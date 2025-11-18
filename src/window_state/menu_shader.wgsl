struct VertexInput {
    @location(0) position: vec2<f32>,
};

struct InstanceInput {
    @location(1) position: vec2<f32>,
    @location(2) size: vec2<f32>,
    @location(3) color: vec4<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
};

struct VertexUniforms {
    scale_factor: vec2<f32>,
    _padding: vec2<f32>,
};

@group(0) @binding(0)
var<uniform> vertex_uniforms: VertexUniforms;

@vertex
fn vs_main(vertex: VertexInput, instance: InstanceInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = vec4(((vertex.position * instance.size + instance.position) * vertex_uniforms.scale_factor * 2 - 1) * vec2(1.0, -1.0), 0.0, 1.0);
    out.color = instance.color;
    return out;
}



@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return in.color;
}