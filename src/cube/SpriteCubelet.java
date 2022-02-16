package cube;

import static org.lwjgl.opengl.GL20.glUniformMatrix4fv;

import math.Matrix;
import shapes.*;

public class SpriteCubelet extends Sprite {
	private SpriteCube parentCube;

	private Box4 rectX, rectY, rectZ;
	private P2[] texCoordsX, texCoordsY, texCoordsZ;
	private int colorX, colorY, colorZ;

	// animation stuff
	private boolean aRotating = false;
	private RNotation.Axis aAxis;
	private double aAngle;

	public SpriteCubelet(SpriteCube pParentCube, int x, int y, int z, int cx, int cy, int cz) {
		super(pParentCube.getProgram());
		parentCube = pParentCube;

		float t = 1f / 3f;
		float limit = 1.0f + SpriteCube.SPACE;

		if (z != 1) {

			float z1 = limit * (z - 1);
			float x1 = -limit + (x + z / 2) * SpriteCube.INCREMENT + SpriteCube.SPACE * x;
			float y1 = limit - y * SpriteCube.INCREMENT - SpriteCube.SPACE * y;
			float x2 = -limit + (x + 1 - z / 2) * SpriteCube.INCREMENT + SpriteCube.SPACE * x;
			float y2 = limit - (y + 1) * SpriteCube.INCREMENT - SpriteCube.SPACE * y;

			rectZ = new Box4(new P3(x1, y1, z1), new P3(x1, y2, z1), new P3(x2, y2, z1), new P3(x2, y1, z1));
			float xc = (z == 0 ? x : 2 - x);
			texCoordsZ = new P2[4];
			texCoordsZ[0] = new P2(xc * t, y * t);
			texCoordsZ[1] = new P2(xc * t, (y + 1) * t);
			texCoordsZ[2] = new P2((xc + 1) * t, (y + 1) * t);
			texCoordsZ[3] = new P2((xc + 1) * t, y * t);
		}

		if (x != 1) {
			float x1 = limit * (x - 1);
			float y1 = limit - y * SpriteCube.INCREMENT - SpriteCube.SPACE * y;
			float y2 = limit - (y + 1) * SpriteCube.INCREMENT - SpriteCube.SPACE * y;
			float z1 = -limit + (z + 1 - x / 2) * SpriteCube.INCREMENT + SpriteCube.SPACE * z;
			float z2 = -limit + (z + x / 2) * SpriteCube.INCREMENT + SpriteCube.SPACE * z;
			rectX = new Box4(new P3(x1, y1, z1), new P3(x1, y2, z1), new P3(x1, y2, z2), new P3(x1, y1, z2));
			float zc = (x == 0 ? 2 - z : z);
			texCoordsX = new P2[4];
			texCoordsX[0] = new P2(zc * t, y * t);
			texCoordsX[1] = new P2(zc * t, (y + 1) * t);
			texCoordsX[2] = new P2((zc + 1) * t, (y + 1) * t);
			texCoordsX[3] = new P2((zc + 1) * t, y * t);
		}

		if (y != 1) {
			float y1 = -limit * (y - 1);
			float x1 = -limit + x * SpriteCube.INCREMENT + SpriteCube.SPACE * x;
			float z1 = -limit + (z + 1 - y / 2) * SpriteCube.INCREMENT + SpriteCube.SPACE * z;
			float x2 = -limit + (x + 1) * SpriteCube.INCREMENT + SpriteCube.SPACE * x;
			float z2 = -limit + (z + y / 2) * SpriteCube.INCREMENT + SpriteCube.SPACE * z;

			rectY = new Box4(new P3(x1, y1, z1), new P3(x1, y1, z2), new P3(x2, y1, z2), new P3(x2, y1, z1));
			float zc = (y == 0 ? 2 - z : z);
			texCoordsY = new P2[4];
			texCoordsY[0] = new P2(x * t, zc * t);
			texCoordsY[1] = new P2(x * t, (zc + 1) * t);
			texCoordsY[2] = new P2((x + 1) * t, (zc + 1) * t);
			texCoordsY[3] = new P2((x + 1) * t, zc * t);
		}
		colorX = cx;
		colorY = cy;
		colorZ = cz;

	}

	public void update() {

		if (aRotating) {
			aAngle -= Math.signum(aAngle) * SpriteCube.MOVE_INC;
			if (Math.abs(aAngle) + SpriteCube.MOVE_INC <= 0) {
				aAngle = 0.0f;
				aAxis = null;
				aRotating = false;
			}
		}

	}

	public void draw_rects() {
		if (colorX != 0)
			rectX.draw(getProgram(), parentCube.getBuffer(), parentCube.getTextures()[colorX], texCoordsX);

		if (colorY != 0)
			rectY.draw(getProgram(), parentCube.getBuffer(), parentCube.getTextures()[colorY], texCoordsY);

		if (colorZ != 0)
			rectZ.draw(getProgram(), parentCube.getBuffer(), parentCube.getTextures()[colorZ], texCoordsZ);
	}

	public void draw() {
		if (aRotating) {
			float[][] matrix = null;
			switch (aAxis) {
			case X:
				matrix = Matrix.multiply(Matrix.loadRotationX(aAngle), parentCube.getMatrix());
				break;
			case Y:
				matrix = Matrix.multiply(Matrix.loadRotationY(aAngle), parentCube.getMatrix());
				break;
			case Z:
				matrix = Matrix.multiply(Matrix.loadRotationZ(aAngle), parentCube.getMatrix());
				break;
			}
			int uniform = getProgram().getUniformLocation("matrix");
			glUniformMatrix4fv(uniform, false, Matrix.toMat4(matrix));
			draw_rects();
			glUniformMatrix4fv(uniform, false, parentCube.getM4());
		} else
			draw_rects();

	}

	public void rotateX(int direction) {
		if (parentCube.getAnimation()) {
			aRotating = true;
			aAxis = RNotation.Axis.X;
			aAngle = direction * 90;
		}
		if (colorX != 0)
			rectX.rotateX(direction);
		if (colorY != 0)
			rectY.rotateX(direction);
		if (colorZ != 0)
			rectZ.rotateX(direction);
	}

	public void rotateY(int direction) {
		if (parentCube.getAnimation()) {
			aRotating = true;
			aAxis = RNotation.Axis.Y;
			aAngle = direction * 90;
		}
		if (colorX != 0)
			rectX.rotateY(direction);
		if (colorY != 0)
			rectY.rotateY(direction);
		if (colorZ != 0)
			rectZ.rotateY(direction);
	}

	public void rotateZ(int direction) {
		if (parentCube.getAnimation()) {
			aRotating = true;
			aAxis = RNotation.Axis.Z;
			aAngle = direction * 90;
		}
		if (colorX != 0)
			rectX.rotateZ(direction);
		if (colorY != 0)
			rectY.rotateZ(direction);
		if (colorZ != 0)
			rectZ.rotateZ(direction);
	}
}
