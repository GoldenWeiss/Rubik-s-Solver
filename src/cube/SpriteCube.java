package cube;

import static org.lwjgl.opengl.GL20.glUniformMatrix4fv;

import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.List;

import main.App;
import math.Matrix;
import shaders.*;

public class SpriteCube extends Sprite {
	
	public static final float SPACE = 0.02f;
	public static final float INCREMENT = 2f / 3f;
	public static final float RESIZE = ( 1f - SPACE / 2f ) * 0.5f;
	
	public static final long MOVE_TIME = 300; // bug en bas de 300 avec animation
	public static final int MOVE_INC = (int) (90 * 60 / MOVE_TIME);
	
	private VertexBuffer buf;
	private Texture[] texs;
	private SpriteCubelet[][][] cubelets = new SpriteCubelet[3][3][3];
	private double angle;
	private float[][] matrix;
	private FloatBuffer m4;
	private boolean animation;
	private List<RNotation> queue;
	private long lastTime;
	private int[][][][] state;
	
	public SpriteCube(GLSLProgram pProgram, VertexBuffer pBuf, Texture[] pTexs, int[][][][] cube) {
		super(pProgram);
		buf = pBuf;
		texs = pTexs;
		angle = 0;
		matrix = Matrix.loadIdentity(4);
		animation = false;
		queue = new ArrayList<RNotation>();
		lastTime = 0;
		loadState(cube);
		
	}

	public void loadState(int[][][][] cube) {
		state = cube;
		for (int x = 0; x < 3; x++) {
			for (int y = 0; y < 3; y++) {
				for (int z = 0; z < 3; z++) {
					int[] colors = cube[x][y][z];
					cubelets[x][y][z] = new SpriteCubelet(this, x, y, z, colors[0], colors[1], colors[2]);
				}
			}
		}
	}
	
	public int[][][][] getState() {
		return Cube.copy(state);
	}
	

	public void  update() {
		angle += 0.5;
		
		if (!queue.isEmpty() && System.currentTimeMillis() - lastTime > SpriteCube.MOVE_TIME) {
			lastTime = System.currentTimeMillis();
			rotate(queue.get(0));
			queue.remove(0);
		}
		for (int x = 0; x < 3; x++) {
			for (int y = 0; y < 3; y++) {
				for (int z = 0; z < 3; z++) {
					cubelets[x][y][z].update();
				}
			}

		}
		
	}

	public void draw() {
		matrix = Matrix.multiply(App.vmatrix, Matrix.loadScalar(4, (SpriteCube.RESIZE)));
		matrix[3][3] = 1;
		matrix = Matrix.multiply(matrix, Matrix.multiply(Matrix.loadRotationZ(angle), Matrix.loadRotationX(angle)));
		 // Pour fonctionner avec opengl.
		
		m4 = Matrix.toMat4(matrix);
		glUniformMatrix4fv(getProgram().getUniformLocation("matrix"), false, m4);

		
		for (int x = 0; x < 3; x++) {
			for (int y = 0; y < 3; y++) {
				for (int z = 0; z < 3; z++) {
					cubelets[x][y][z].draw();
				}
			}
		}
		glUniformMatrix4fv(getProgram().getUniformLocation("matrix"), false, App.vm4);
	}

	public VertexBuffer getBuffer() {
		return buf;
	}
	public float[][] getMatrix() {
		return matrix;
	}
	
	public FloatBuffer getM4() {
		return m4;
	}
	
	public Texture[] getTextures() {
		return texs;
	}
	public int historyId() {
		return queue.size();
	}

	public void setTextures(Texture[] pTexs) {
		texs = pTexs;
	}
	public boolean getAnimation() {
		return animation;
	}
	public void setAnimation(boolean value) {
		animation = value;
	}
	
	/**
	 *  La méthode à utiliser pour faire tourner le cube avec animation
	 * @param pNotation
	 */
	public void  addRotation(RNotation pNotation) {
		queue.add(pNotation);
	}
	
	/**
	 * Pour le pathfinding
	 * @param pNotations
	 */
	public void setRotations(List<RNotation> pNotations) {
		queue = pNotations;
	}
	
	/**
	 *  à utilier si vous voulez un scramble au départ déja fait
	 * @param pNotation
	 */
	public void rotate(RNotation pNotation) { 
		state = Cube.rotate(state, pNotation);
		
		switch (pNotation) {
		case L:
			xRotation(0, 1);
			break;
		case Li:
			xRotation(0, -1);
			break;
		case R:
			xRotation(2, -1);
			break;
		case Ri:
			xRotation(2, 1);
			break;
		case U:
			yRotation(0, -1);
			break;
		case Ui:
			yRotation(0, 1);
			break;
		case D:
			yRotation(2, -1);
			break;
		case Di:
			yRotation(2, 1);
			break;
		case B:
			zRotation(2, -1);
			break;
		case Bi:
			zRotation(2, 1);
			break;
		case F:
			zRotation(0, 1);
			break;
		case Fi:
			zRotation(0, -1);
			break;
		default:
			return;

		}
	}
	
	public void swapCubelets(int x1, int y1, int z1, int x2, int y2, int z2) {
		SpriteCubelet tcbt = cubelets[x2][y2][z2];
		cubelets[x2][y2][z2] = cubelets[x1][y1][z1];
		cubelets[x1][y1][z1] = tcbt;
	}
	
	public void xRotation(int x, int direction) {
		cubelets[x][1][1].rotateX(direction);
		for (int i = 0; i < 2; i++) {
			cubelets[x][i][0].rotateX(direction);
			cubelets[x][0][2 - i].rotateX(direction);
			cubelets[x][2 - i][2].rotateX(direction);
			cubelets[x][2][i].rotateX(direction);
			swapCubelets(x, i, 0, x, 2 - i, 2);
			switch (direction) {
			case 1:
				swapCubelets(x, 2 - i, 2, x, 0, 2 - i);
				swapCubelets(x, i, 0, x, 2, i);
				break;
			case -1:
				swapCubelets(x, 2 - i, 2, x, 2, i);
				swapCubelets(x, i, 0, x, 0, 2 - i);
				break;
			}
		}
	}

	public void yRotation(int y, int direction) {
		cubelets[1][y][1].rotateY(direction);
		for (int i = 0; i < 2; i++) {
			cubelets[i][y][0].rotateY(direction);
			cubelets[0][y][2 - i].rotateY(direction);
			cubelets[2 - i][y][2].rotateY(direction);
			cubelets[2][y][i].rotateY(direction);
			swapCubelets(i, y, 0, 2 - i, y, 2);
			switch (direction) {
			case 1:
				swapCubelets(2 - i, y, 2, 0, y, 2 - i);
				swapCubelets(i, y, 0, 2, y, i);
				break;
			case -1:
				swapCubelets(2 - i, y, 2, 2, y, i);
				swapCubelets(i, y, 0, 0, y, 2 - i);
				break;
			}
		}
	}

	public void zRotation(int z, int direction) {
		cubelets[1][1][z].rotateZ(direction);
		for (int i = 0; i < 2; i++) {
			cubelets[i][0][z].rotateZ(direction);
			cubelets[0][2 - i][z].rotateZ(direction);
			cubelets[2 - i][2][z].rotateZ(direction);
			cubelets[2][i][z].rotateZ(direction);
			swapCubelets(i, 0, z, 2 - i, 2, z);
			switch (direction) {
			case 1:
				swapCubelets(2 - i, 2, z, 0, 2 - i, z);
				swapCubelets(i, 0, z, 2, i, z);
				break;
			case -1:
				swapCubelets(2 - i, 2, z, 2, i, z);
				swapCubelets(i, 0, z, 0, 2 - i, z);
				break;
			}
		}
	}

}
