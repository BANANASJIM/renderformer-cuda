#pragma once
#include <cmath>
#include <cstring>
#include <algorithm>

// Trackball camera orbiting a center point (Z-up world)
// c2w convention: Blender (-Z forward, +Y up, +X right) in camera-local frame
struct Camera {
    float theta = 0;          // azimuth in XY plane (radians)
    float phi = M_PI_2;       // polar angle from +Z axis
    float radius = 1.75f;
    float center[3] = {};
    float fov_deg = 39.6f;

    void orbit(float dtheta, float dphi) {
        theta += dtheta;
        phi = std::clamp(phi + dphi, 0.02f, (float)M_PI - 0.02f);
    }

    void zoom(float factor) { radius = std::clamp(radius * factor, 0.3f, 10.0f); }
    float fov_rad() const { return fov_deg * (float)(M_PI / 180.0); }

    void eye(float* out) const {
        float sp = sinf(phi), cp = cosf(phi);
        out[0] = center[0] + radius * sp * cosf(theta);
        out[1] = center[1] + radius * sp * sinf(theta);
        out[2] = center[2] + radius * cp;
    }

    // Build 4x4 camera-to-world matrix (row-major, Blender convention)
    void get_c2w(float* m) const {
        float e[3]; eye(e);

        // forward = normalize(center - eye)
        float f[3] = {center[0]-e[0], center[1]-e[1], center[2]-e[2]};
        float fl = sqrtf(f[0]*f[0] + f[1]*f[1] + f[2]*f[2]);
        f[0]/=fl; f[1]/=fl; f[2]/=fl;

        // right = normalize(forward × world_up), world_up = (0,0,1)
        float r[3] = {f[1] - 0.0f, -f[0], 0.0f};
        float rl = sqrtf(r[0]*r[0] + r[1]*r[1]);
        if (rl < 1e-6f) { r[0]=1; r[1]=0; r[2]=0; rl=1; }
        r[0]/=rl; r[1]/=rl; r[2]/=rl;

        // up = right × forward
        float u[3] = {r[1]*f[2]-r[2]*f[1], r[2]*f[0]-r[0]*f[2], r[0]*f[1]-r[1]*f[0]};

        // Blender c2w: columns = [right, up, -forward, position]
        std::memset(m, 0, 64);
        m[0]=r[0]; m[1]=u[0]; m[2]=-f[0]; m[3]=e[0];
        m[4]=r[1]; m[5]=u[1]; m[6]=-f[1]; m[7]=e[1];
        m[8]=r[2]; m[9]=u[2]; m[10]=-f[2]; m[11]=e[2];
        m[15]=1;
    }
};
