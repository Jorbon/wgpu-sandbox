

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
};

struct InstanceInput {
    @location(5) model_transform_0: vec4<f32>,
    @location(6) model_transform_1: vec4<f32>,
    @location(7) model_transform_2: vec4<f32>,
    @location(8) model_transform_3: vec4<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) normal: vec3<f32>,
};

struct VertexUniforms {
    camera_transform: mat4x4<f32>,
};

@group(0) @binding(0)
var<uniform> vertex_uniforms: VertexUniforms;

@vertex
fn vs_main(model: VertexInput, instance: InstanceInput) -> VertexOutput {
    let model_transform = mat4x4<f32>(
        instance.model_transform_0,
        instance.model_transform_1,
        instance.model_transform_2,
        instance.model_transform_3,
    );
    
    var out: VertexOutput;
    out.clip_position = vertex_uniforms.camera_transform * model_transform * vec4<f32>(model.position, 1.0);
    out.normal = model.normal;
    return out;
}


@group(1) @binding(0)
var t: texture_2d<f32>;
@group(1) @binding(1)
var s: sampler;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return textureSample(t, s, (1.0 - in.normal.xy) * 0.5);
}

