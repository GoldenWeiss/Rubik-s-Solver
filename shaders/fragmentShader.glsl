#version 150 core
in vec3 Color;
in vec2 Texcoord;
out vec4 outColor;
uniform sampler2D tex;
void main()
{
	if (gl_FrontFacing) {
		outColor = texture(tex, Texcoord) * vec4(Color, 1.0);
	} else {
		outColor = vec4(0, 0, 0, 1);
	}
}
