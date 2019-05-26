#version 450

layout(set = 1, binding = 0) uniform sampler2D tex;

layout(location = 0) in vec2 tex_coords;

layout(location = 0) out vec4 frag_color;

void main() {
  frag_color = texture(tex, tex_coords);
}
