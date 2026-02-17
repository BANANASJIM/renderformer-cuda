#include "model/renderformer.h"
#include "pipeline/render_session.h"
#include "io/hdf5_reader.h"
#include "app/camera.h"
#include <GLFW/glfw3.h>
#include <cstdio>
#include <cmath>
#include <string>
#include <vector>

// ---- GL state ----
static GLuint tex_id = 0;
static int render_res = 512; // render resolution (texture size)

static void init_gl_texture(int w, int h) {
    glGenTextures(1, &tex_id);
    glBindTexture(GL_TEXTURE_2D, tex_id);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, w, h, 0, GL_RGB, GL_FLOAT, nullptr);
}

// CHW fp32 HDR → HWC gamma-corrected → GL texture upload
static void tonemap_upload(const float* chw, int w, int h) {
    // Flip vertically: OpenGL texture (0,0) = bottom-left, our data row 0 = top
    std::vector<float> rgb(w * h * 3);
    for (int y = 0; y < h; y++) {
        int src_y = y;
        int dst_y = h - 1 - y; // flip
        for (int x = 0; x < w; x++)
            for (int c = 0; c < 3; c++) {
                float v = chw[c * h * w + src_y * w + x];
                v = fmaxf(v, 0.0f);
                v = powf(fminf(v, 10.0f), 1.0f / 2.2f);
                rgb[(dst_y * w + x) * 3 + c] = v;
            }
    }
    glBindTexture(GL_TEXTURE_2D, tex_id);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w, h, GL_RGB, GL_FLOAT, rgb.data());
}

static void draw_fullscreen_quad(GLFWwindow* win) {
    int fb_w, fb_h;
    glfwGetFramebufferSize(win, &fb_w, &fb_h);

    // Letterbox: square viewport centered in potentially non-square framebuffer
    int sz = std::min(fb_w, fb_h);
    glViewport((fb_w - sz) / 2, (fb_h - sz) / 2, sz, sz);

    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, tex_id);
    glBegin(GL_QUADS);
    glTexCoord2f(0, 0); glVertex2f(-1, -1);
    glTexCoord2f(1, 0); glVertex2f( 1, -1);
    glTexCoord2f(1, 1); glVertex2f( 1,  1);
    glTexCoord2f(0, 1); glVertex2f(-1,  1);
    glEnd();
}

// ---- Input state ----
static Camera cam;
static bool camera_dirty = true;
static double last_mx = 0, last_my = 0;
static bool dragging = false;

static void mouse_button_cb(GLFWwindow*, int button, int action, int) {
    if (button == GLFW_MOUSE_BUTTON_LEFT) dragging = (action == GLFW_PRESS);
}

static void cursor_pos_cb(GLFWwindow*, double x, double y) {
    if (dragging) {
        cam.orbit(-(float)(x - last_mx) * 0.005f, -(float)(y - last_my) * 0.005f);
        camera_dirty = true;
    }
    last_mx = x; last_my = y;
}

static void scroll_cb(GLFWwindow*, double, double yoff) {
    cam.zoom(1.0f - 0.1f * (float)yoff);
    camera_dirty = true;
}

static void key_cb(GLFWwindow* win, int key, int, int action, int) {
    if (action != GLFW_PRESS && action != GLFW_REPEAT) return;
    float step = 0.05f;
    switch (key) {
        case GLFW_KEY_ESCAPE: case GLFW_KEY_Q:
            glfwSetWindowShouldClose(win, 1); break;
        case GLFW_KEY_R: cam = Camera{}; camera_dirty = true; break;
        case GLFW_KEY_LEFT:  cam.orbit( step, 0); camera_dirty = true; break;
        case GLFW_KEY_RIGHT: cam.orbit(-step, 0); camera_dirty = true; break;
        case GLFW_KEY_UP:    cam.orbit(0,  step); camera_dirty = true; break;
        case GLFW_KEY_DOWN:  cam.orbit(0, -step); camera_dirty = true; break;
    }
}

// Extract Z-up spherical coords from c2w position
static void init_camera_from_c2w(Camera& cam, const float* c2w) {
    float px = c2w[3], py = c2w[7], pz = c2w[11];
    cam.radius = sqrtf(px*px + py*py + pz*pz);
    if (cam.radius < 0.01f) cam.radius = 1.75f;
    cam.phi = acosf(std::clamp(pz / cam.radius, -1.0f, 1.0f));
    cam.theta = atan2f(py, px);
}

int main(int argc, char** argv) {
    std::string model_path, scene_path;
    int img_res = 512;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--model" && i+1 < argc) model_path = argv[++i];
        else if (arg == "--scene" && i+1 < argc) scene_path = argv[++i];
        else if (arg == "--res" && i+1 < argc) img_res = std::stoi(argv[++i]);
    }
    if (scene_path.empty() || model_path.empty()) {
        fprintf(stderr, "Usage: renderformer_app --scene <h5> --model <safetensors> [--res <int>]\n");
        return 1;
    }

    render_res = img_res;

    printf("Loading scene: %s\n", scene_path.c_str()); fflush(stdout);
    auto scene = rf::Scene::load(scene_path);
    printf("Loading model: %s\n", model_path.c_str()); fflush(stdout);
    auto sf = rf::SafeTensorsFile::load(model_path);
    rf::RenderFormerModel model;
    model.load(sf);

    // Initialize camera from scene's first view
    if (scene.num_cameras > 0) {
        std::vector<float> fov(scene.num_cameras), c2w(scene.num_cameras * 16);
        cudaMemcpy(fov.data(), scene.fov().fp32(), scene.num_cameras * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(c2w.data(), scene.c2w().fp32(), scene.num_cameras * 16 * sizeof(float), cudaMemcpyDeviceToHost);
        cam.fov_deg = fov[0];
        init_camera_from_c2w(cam, c2w.data());
        printf("Camera: theta=%.2f phi=%.2f r=%.2f fov=%.1f\n",
               cam.theta, cam.phi, cam.radius, cam.fov_deg); fflush(stdout);
    }

    auto session = rf::RenderSession::create(model, scene, img_res);

    // Init GLFW
    if (!glfwInit()) { fprintf(stderr, "Failed to init GLFW\n"); return 1; }
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    glfwWindowHint(GLFW_FLOATING, GLFW_TRUE);

    auto* win = glfwCreateWindow(img_res, img_res, "RenderFormer Viewer", nullptr, nullptr);
    glfwSetWindowAspectRatio(win, 1, 1);
    if (!win) { fprintf(stderr, "Failed to create window\n"); glfwTerminate(); return 1; }
    glfwMakeContextCurrent(win);
    glfwSwapInterval(0);

    int fb_w, fb_h;
    glfwGetFramebufferSize(win, &fb_w, &fb_h);
    printf("Window: %dx%d, Framebuffer: %dx%d\n", img_res, img_res, fb_w, fb_h);
    fflush(stdout);

    glfwSetMouseButtonCallback(win, mouse_button_cb);
    glfwSetCursorPosCallback(win, cursor_pos_cb);
    glfwSetScrollCallback(win, scroll_cb);
    glfwSetKeyCallback(win, key_cb);

    init_gl_texture(img_res, img_res);

    // First frame (synchronous)
    float c2w[16];
    cam.get_c2w(c2w);
    const float* frame = session.render_frame(c2w, cam.fov_rad());
    printf("First frame: %.1f ms\n", session.last_frame_ms()); fflush(stdout);

    std::vector<float> h_chw(3 * img_res * img_res);
    cudaMemcpy(h_chw.data(), frame, h_chw.size() * sizeof(float), cudaMemcpyDeviceToHost);
    tonemap_upload(h_chw.data(), img_res, img_res);
    camera_dirty = false;

    bool rendering = false;
    int frame_count = 0;
    double fps_time = glfwGetTime();

    while (!glfwWindowShouldClose(win)) {
        glfwPollEvents();

        if (camera_dirty && !rendering) {
            cam.get_c2w(c2w);
            session.render_frame_async(c2w, cam.fov_rad());
            rendering = true;
            camera_dirty = false;
        }

        if (rendering && session.is_frame_ready()) {
            frame = session.get_frame_result();
            cudaMemcpy(h_chw.data(), frame, h_chw.size() * sizeof(float), cudaMemcpyDeviceToHost);
            tonemap_upload(h_chw.data(), img_res, img_res);
            rendering = false;
            frame_count++;
        }

        glClear(GL_COLOR_BUFFER_BIT);
        draw_fullscreen_quad(win);

        double now = glfwGetTime();
        if (now - fps_time >= 1.0) {
            char title[128];
            snprintf(title, sizeof(title), "RenderFormer | %.1f ms/frame | %d FPS",
                     session.last_frame_ms(), frame_count);
            glfwSetWindowTitle(win, title);
            frame_count = 0;
            fps_time = now;
        }

        glfwSwapBuffers(win);
    }

    glfwDestroyWindow(win);
    glfwTerminate();
    return 0;
}
