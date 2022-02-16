package shapes;

import static org.lwjgl.opengl.GL30.*;

import shaders.*;

public class Box4 {
	
	public P3 p1, p2, p3, p4;
	public Box4(P3 _p1, P3 _p2, P3 _p3, P3 _p4) {
		p1 = _p1;
		p2 = _p2;
		p3 = _p3;
		p4 = _p4;
	}
	
	
	public void draw(GLSLProgram program, VertexBuffer buf, Texture tex) {
		buf.setSubData(0 , p1.getCoords());
		buf.setSubData(32, p2.getCoords());
		buf.setSubData(64, p3.getCoords());
		buf.setSubData(96, p4.getCoords());
		
		
		glUniform1i(program.getUniformLocation("tex"), tex.getId());
		
		tex.bind();
			glDrawArrays(GL_QUADS, 0, 4);
		tex.unbind();
	}
	
	public void draw(GLSLProgram program, VertexBuffer buf, Texture tex, P2[] texCoords) {
		buf.setSubData(24, texCoords[0].getCoords());
		buf.setSubData(56, texCoords[1].getCoords());
		buf.setSubData(88, texCoords[2].getCoords());
		buf.setSubData(120, texCoords[3].getCoords());
		
		draw(program, buf, tex);
	}

	public void rotateX(int direction) {
		p1.rotateX(direction);
		p2.rotateX(direction);
		p3.rotateX(direction);
		p4.rotateX(direction);
	}
	public void rotateY(int direction) {
		p1.rotateY(direction);
		p2.rotateY(direction);
		p3.rotateY(direction);
		p4.rotateY(direction);
		
	}
	public void rotateZ(int direction) {
		p1.rotateZ(direction);
		p2.rotateZ(direction);
		p3.rotateZ(direction);
		p4.rotateZ(direction);
		
	}
}
