package cube;

import math.Matrix;

public class Cube {
	public static final int FORMAT_SIZE_CUBE3 = 6 * 3 * 3 * 3 * 2;// (3*3*3*3/4+3*3*3/2+3*3);
	public static final int GOD_NUMBER = 26;

	public static int N_INSTANCES = 0;

	private Cube() {
	}

	public static boolean solved(int[][][][] cube) {
		boolean zsolved = true, ysolved = true, xsolved = true;

		for (int x = 0; x < 3; x++)
			for (int y = 0; y < 3; y++)
				for (int z = 0; z < 3 && zsolved && ysolved && xsolved; z++) {
					if (z != 1)
						zsolved = zsolved && (cube[x][y][z][2] == cube[Math.max(0, x - 1)][Math.max(0, y - 1)][z][2]);
					if (y != 1)
						ysolved = ysolved && (cube[x][y][z][1] == cube[Math.max(0, x - 1)][y][Math.max(0, z - 1)][1]);
					if (x != 1)
						xsolved = xsolved && (cube[x][y][z][0] == cube[x][Math.max(0, y - 1)][Math.max(0, z - 1)][0]);
				}
		return zsolved && ysolved && xsolved;
	}

	public static boolean same(int[][][][] cube1, int[][][][] cube2) {
		for (int x = 0; x < 3; x++)
			for (int y = 0; y < 3; y++)
				for (int z = 0; z < 3; z++)
					for (int c = 0; c < 3; c++)
						if (cube1[x][y][z][c] != cube2[x][y][z][c])
							return false;
		return true;
	}

	public static int[][][][] copy(int[][][][] cube) {
		if (++N_INSTANCES % 100_000 == 0) {
			try {
				Thread.currentThread().sleep(100);
			} catch (InterruptedException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}

		int[][][][] ary = new int[3][3][3][3];
		for (int x = 0; x < 3; x++)
			for (int y = 0; y < 3; y++)
				for (int z = 0; z < 3; z++)
					for (int c = 0; c < 3; c++)
						ary[x][y][z][c] = cube[x][y][z][c];

		return ary;
	}

	public static int[][][][] loadSolved() {
		int[][][][] ary = new int[3][3][3][3];
		for (int x = 0; x < 3; x++) {
			for (int y = 0; y < 3; y++) {
				for (int z = 0; z < 3; z++) {
					ary[x][y][z][0] = baseColorX(x);
					ary[x][y][z][1] = baseColorY(y);
					ary[x][y][z][2] = baseColorZ(z);
				}
			}
		}
		return ary;
	}

	public static void swapCubelets(int[][][][] cube, int x1, int y1, int z1, int x2, int y2, int z2) {
		int[] tcbt = cube[x2][y2][z2];
		cube[x2][y2][z2] = cube[x1][y1][z1];
		cube[x1][y1][z1] = tcbt;
	}

	private static void swapColors(int[] cbt, int i1, int i2) {
		int tc = cbt[i2];
		cbt[i2] = cbt[i1];
		cbt[i1] = tc;
	}

	private static void xRotate(int[][][][] cube, int x, int d) {
		for (int i = 0; i < 2; i++) {
			swapColors(cube[x][i][0], 1, 2);
			swapColors(cube[x][2 - i][2], 1, 2);
			swapColors(cube[x][0][2 - i], 1, 2);
			swapColors(cube[x][2][i], 1, 2);

			swapCubelets(cube, x, i, 0, x, 2 - i, 2);
			switch (d) {
			case 1:
				swapCubelets(cube, x, 2 - i, 2, x, 0, 2 - i);
				swapCubelets(cube, x, i, 0, x, 2, i);
				break;
			case -1:
				swapCubelets(cube, x, 2 - i, 2, x, 2, i);
				swapCubelets(cube, x, i, 0, x, 0, 2 - i);
				break;
			}
		}
	}

	private static void yRotate(int[][][][] cube, int y, int d) {
		for (int i = 0; i < 2; i++) {
			swapColors(cube[i][y][0], 0, 2);
			swapColors(cube[2 - i][y][2], 0, 2);
			swapColors(cube[0][y][2 - i], 0, 2);
			swapColors(cube[2][y][i], 0, 2);

			swapCubelets(cube, i, y, 0, 2 - i, y, 2);
			switch (d) {
			case 1:
				swapCubelets(cube, 2 - i, y, 2, 0, y, 2 - i);
				swapCubelets(cube, i, y, 0, 2, y, i);
				break;
			case -1:
				swapCubelets(cube, 2 - i, y, 2, 2, y, i);
				swapCubelets(cube, i, y, 0, 0, y, 2 - i);
				break;
			}
		}
	}

	private static void zRotate(int[][][][] cube, int z, int d) {
		for (int i = 0; i < 2; i++) {
			swapColors(cube[i][0][z], 0, 1);
			swapColors(cube[2 - i][2][z], 0, 1);
			swapColors(cube[0][2 - i][z], 0, 1);
			swapColors(cube[2][i][z], 0, 1);

			swapCubelets(cube, i, 0, z, 2 - i, 2, z);
			switch (d) {
			case 1:
				swapCubelets(cube, 2 - i, 2, z, 0, 2 - i, z);
				swapCubelets(cube, i, 0, z, 2, i, z);
				break;
			case -1:
				swapCubelets(cube, 2 - i, 2, z, 2, i, z);
				swapCubelets(cube, i, 0, z, 0, 2 - i, z);
				break;
			}

		}
	}

	public static void staticRotate(int[][][][] cube, RNotation move) {

		switch (move) {
		case L:
			xRotate(cube, 0, 1);
			break;
		case Li:
			xRotate(cube, 0, -1);
			break;
		case R:
			xRotate(cube, 2, -1);
			break;
		case Ri:
			xRotate(cube, 2, 1);
			break;
		case U:
			yRotate(cube, 0, -1);
			break;
		case Ui:
			yRotate(cube, 0, 1);
			break;
		case D:
			yRotate(cube, 2, 1);
			break;
		case Di:
			yRotate(cube, 2, -1);
			break;
		case B:
			zRotate(cube, 2, -1);
			break;
		case Bi:
			zRotate(cube, 2, 1);
			break;
		case F:
			zRotate(cube, 0, 1);
			break;
		case Fi:
			zRotate(cube, 0, -1);
			break;
		}
	}

	public static int[][][][] scramble(int[][][][] cube, int depth) {
		int[][][][] copy = Cube.copy(cube);

		RNotation cache = null;
		RNotation temp;

		for (int i = 0; i < depth; i++) {

			do
				temp = RNotation.fromId(RNotation.randomRotation());
			while (RNotation.invertedRotation(temp) == cache);

			Cube.staticRotate(copy, cache = temp);
		}
		return copy;
	}

	public static int[][][][] rotate(int[][][][] cube, RNotation move) {
		int[][][][] copy = Cube.copy(cube);
		staticRotate(copy, move);
		return copy;
	}

	public static int baseColorX(int x) {
		switch (x) {
		case 0:
			return 4;
		case 2:
			return 3;
		}
		return 0;
	}

	public static int baseColorY(int y) {
		switch (y) {
		case 0:
			return 5;
		case 2:
			return 6;
		}
		return 0;
	}

	public static int baseColorZ(int z) {
		switch (z) {
		case 0:
			return 1;
		case 2:
			return 2;
		}
		return 0;
	}

	/**
	 * Map 4d int array to 1d float array. The cube3 format is a matrix data of the
	 * cube stored in neural network (one-hot encoding) TODO REFAIRE LE FORMAT IL Y
	 * A QUELQUES POINTS BIZARRES ex : [1.0 0.0 0.0 1.0 0.0 0.0]
	 * 
	 * @param cube
	 * @return
	 */
	public static float[] toCube3(int[][][][] cube) {

		float[] ary = new float[FORMAT_SIZE_CUBE3];

		for (int x = 0; x < 3; x++)
			for (int y = 0; y < 3; y++)
				for (int z = 0; z < 3; z++) {

					if (z != 1)
						ary[9 * 6 * (z / 2) + 0 + 6 * (3 * x + y) + cube[x][y][z][2] - 1] = 1;

					if (y != 1)
						ary[9 * 6 * (y / 2) + 6 * 2 * 9 + 6 * (3 * x + z) + cube[x][y][z][1] - 1] = 1;

					if (x != 1)
						ary[9 * 6 * (x / 2) + 6 * 4 * 9 + 6 * (3 * z + y) + cube[x][y][z][0] - 1] = 1;

				}
		return ary;
	}

	public static float[][][] toChannel(int[][][][] cube) {
		return toChannel4(cube);
	}

	public static float[][][] toChannel4(int[][][][] cube) {
		float[][][] ary = new float[1][16][5];

		// front
		for (int y = 0; y < 3; y++) {
			for (int x = 0; x < 3; x++)
				ary[0][2 - x][y + 1] = cube[x][y][0][2];
			ary[0][3][y + 1] = cube[0][y][0][0];
		}
		for (int x = 0; x < 3; x++) {
			ary[0][2 - x][0] = cube[x][0][0][1];
			ary[0][2 - x][4] = cube[x][2][0][1];
		}

		// right
		for (int y = 0; y < 3; y++) {
			for (int z = 0; z < 3; z++)
				ary[0][4+ 2 - z][y + 1] = cube[2][y][z][0];
			ary[0][4+ 3][y + 1] = cube[2][y][0][2];
		}
		for (int z = 0; z < 3; z++) {
			ary[0][4+ 2 - z][0] = cube[2][0][z][1];
			ary[0][4+ 2 - z][4] = cube[2][2][z][1];
		}

		// back
		for (int y = 0; y < 3; y++) {
			for (int x = 0; x < 3; x++)
				ary[0][8+ x][y + 1] = cube[x][y][2][2];
			ary[0][8+ 3][y + 1] = cube[2][y][2][0];
		}
		for (int x = 0; x < 3; x++) {
			ary[0][8+ x][0] = cube[x][0][2][1];
			ary[0][8+ x][4] = cube[x][2][2][1];
		}

		// left
		for (int y = 0; y < 3; y++) {
			for (int z = 0; z < 3; z++)
				ary[0][12+ z][y + 1] = cube[0][y][z][0];
			ary[0][12+ 3][y + 1] = cube[0][y][2][2];
		}
		for (int z = 0; z < 3; z++) {
			ary[0][12+ z][0] = cube[0][0][z][1];
			ary[0][12+ z][4] = cube[0][2][z][1];
		}
		
		//ary[5] = ary[4] = ary[3] = ary[2] = ary[1] = ary[0];
		
		return ary;
	}

	// shape[54, 6] 1 channel
	public static float[][][] toChannel3(int[][][][] cube) {
		float[][][] ary = new float[1][54][6];
		for (int x = 0; x < 3; x++)
			for (int y = 0; y < 3; y++)
				for (int z = 0; z < 3; z++) {
					if (z != 1)
						ary[0][36 + 9 * (z / 2) + 3 * (z == 0 ? x : 2 - x) + y][cube[x][y][z][2] - 1] = 1;
					if (y != 1)
						ary[0][18 + 9 * (y / 2) + 3 * (y == 0 ? 2 - z : z) + x][cube[x][y][z][1] - 1] = 1;
					if (x != 1)
						ary[0][9 * (x / 2) + 3 * (x == 0 ? 2 - z : z) + y][cube[x][y][z][0] - 1] = 1;
				}

		return ary;
	}

	// shape[9, 6 * 6] 1 channel
	public static float[][][] toChannel2(int[][][][] cube) {
		float[][][] ary = new float[1][9][6 * 6];

		for (int x = 0; x < 3; x++)
			for (int y = 0; y < 3; y++)
				for (int z = 0; z < 3; z++) {

					if (z != 1)
						ary[0][y][3 * 6 * (z / 2) + 6 * (z == 0 ? x : 2 - x) + cube[x][y][z][2] - 1] = 1;
					if (y != 1)
						ary[0][3 + x][3 * 6 * (y / 2) + 6 * (y == 0 ? 2 - z : z) + cube[x][y][z][1] - 1] = 1;
					if (x != 1)
						ary[0][6 + y][3 * 6 * (x / 2) + 6 * (x == 0 ? 2 - z : z) + cube[x][y][z][0] - 1] = 1;
				}

		return ary;
	}

	// shape [3, 3 * 6] , color 6 channel notation
	public static float[][][] toChannel1(int[][][][] cube) {

		float[][][] ary = new float[6][3][6 * 3];

		for (int x = 0; x < 3; x++)
			for (int y = 0; y < 3; y++)
				for (int z = 0; z < 3; z++) {

					if (z != 1)
						ary[z / 2][y][6 * (z == 0 ? x : 2 - x) + cube[x][y][z][2] - 1] = 1;
					if (y != 1)
						ary[2 + y / 2][x][6 * (y == 0 ? 2 - z : z) + cube[x][y][z][1] - 1] = 1;
					if (x != 1)
						ary[4 + x / 2][y][6 * (x == 0 ? 2 - z : z) + cube[x][y][z][0] - 1] = 1;
				}

		return ary;
	}

	public static void print(int[][][][] cube) {

		float[][][] channel = Cube.toChannel(cube);

		for (int i = 0; i < 6; i++) {
			System.out.println("F" + i);
			Matrix.print(channel[i]);
		}
	}
}
