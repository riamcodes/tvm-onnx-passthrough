import onnx
from onnx import helper, TensorProto, OperatorSetIdProto

x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])
y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])

c_tensor = helper.make_tensor(
    name="c",
    data_type=TensorProto.FLOAT,
    dims=[1],
    vals=[2.0],
)

const_node = helper.make_node(
    "Constant",
    inputs=[],
    outputs=["c_out"],
    value=c_tensor,
)

add_node = helper.make_node(
    "Add",
    inputs=["x", "c_out"],
    outputs=["y"],
)

graph = helper.make_graph(
    [const_node, add_node],
    "AddTwoGraph",
    [x],
    [y],
)

opset = OperatorSetIdProto()
opset.version = 13

model = helper.make_model(
    graph,
    producer_name="test-model",
    opset_imports=[opset],
)

# Force an older IR version that your onnxruntime supports
model.ir_version = 10

onnx.save(model, "test_model.onnx")
print("Saved test_model.onnx with IR version", model.ir_version)
