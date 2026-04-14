#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>

namespace py = pybind11;

class TVMModule {
public:
    explicit TVMModule(const std::string& model_path) : model_path_(model_path) {}

    std::string model_path() const {
        return model_path_;
    }

    py::object run(py::dict inputs) const {
        py::module_ ort = py::module_::import("onnxruntime");
        py::object session = ort.attr("InferenceSession")(model_path_);
        py::list output_names;  // empty list = ask for all outputs
        return session.attr("run")(output_names, inputs);
    }

private:
    std::string model_path_;
};

PYBIND11_MODULE(_native, m) {
    m.doc() = "Native TVM-like pass-through extension";

    py::class_<TVMModule>(m, "TVMModule")
        .def(py::init<const std::string&>())
        .def("model_path", &TVMModule::model_path)
        .def("run", &TVMModule::run);
}
