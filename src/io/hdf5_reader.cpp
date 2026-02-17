#include "io/hdf5_reader.h"
#include <hdf5.h>
#include <vector>
#include <stdexcept>

namespace rf {

namespace {

struct H5File {
    hid_t id;
    H5File(const std::string& path) : id(H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT)) {
        if (id < 0) throw std::runtime_error("cannot open HDF5: " + path);
    }
    ~H5File() { H5Fclose(id); }
};

struct H5Dataset {
    hid_t id;
    H5Dataset(hid_t file, const char* name) : id(H5Dopen2(file, name, H5P_DEFAULT)) {
        if (id < 0) throw std::runtime_error(std::string("dataset not found: ") + name);
    }
    ~H5Dataset() { H5Dclose(id); }

    std::vector<hsize_t> shape() const {
        hid_t space = H5Dget_space(id);
        int ndim = H5Sget_simple_extent_ndims(space);
        std::vector<hsize_t> dims(ndim);
        H5Sget_simple_extent_dims(space, dims.data(), nullptr);
        H5Sclose(space);
        return dims;
    }
};

GpuBuffer read_dataset_fp32(hid_t file, const char* name, std::vector<hsize_t>& out_shape) {
    H5Dataset ds(file, name);
    out_shape = ds.shape();
    size_t numel = 1;
    for (auto d : out_shape) numel *= d;

    std::vector<float> host(numel);
    H5Dread(ds.id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, host.data());
    return upload(host.data(), numel * sizeof(float));
}

GpuBuffer read_dataset_fp16(hid_t file, const char* name, std::vector<hsize_t>& out_shape) {
    H5Dataset ds(file, name);
    out_shape = ds.shape();
    size_t numel = 1;
    for (auto d : out_shape) numel *= d;

    hid_t dtype = H5Dget_type(ds.id);
    size_t dt_size = H5Tget_size(dtype);

    if (dt_size == 2) {
        // Native fp16 — use dataset's own type to preserve bit patterns
        std::vector<uint16_t> host(numel);
        H5Dread(ds.id, dtype, H5S_ALL, H5S_ALL, H5P_DEFAULT, host.data());
        H5Tclose(dtype);
        return upload(host.data(), numel * 2);
    } else {
        H5Tclose(dtype);
        // Stored as fp32, convert to fp16 on host
        std::vector<float> host_f32(numel);
        H5Dread(ds.id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, host_f32.data());
        std::vector<uint16_t> host_f16(numel);
        for (size_t i = 0; i < numel; ++i) {
            // Use CUDA's host-side fp16 conversion
            __half h = __float2half(host_f32[i]);
            memcpy(&host_f16[i], &h, 2);
        }
        return upload(host_f16.data(), numel * 2);
    }
}

} // anonymous namespace

Scene Scene::load(const std::string& h5_path) {
    H5File file(h5_path);
    Scene s;

    std::vector<hsize_t> shape;

    s.triangles_buf = read_dataset_fp32(file.id, "triangles", shape);
    s.num_triangles = shape[0];

    s.texture_buf = read_dataset_fp16(file.id, "texture", shape);

    s.vn_buf = read_dataset_fp32(file.id, "vn", shape);

    s.c2w_buf = read_dataset_fp32(file.id, "c2w", shape);
    s.num_cameras = shape[0];

    s.fov_buf = read_dataset_fp32(file.id, "fov", shape);

    return s;
}

} // namespace rf
