package math;

import java.util.Arrays;
import java.util.function.DoubleBinaryOperator;
import java.util.function.DoubleUnaryOperator;
import java.nio.FloatBuffer;
import org.lwjgl.BufferUtils;

import com.aparapi.Kernel;

/**
 * Ne jamais utiliser de constructeur pour cette classe; une matrice est déjà
 * traitée comme un tableau {@code float[][]}. Il ne reste plus qu'à lui
 * appliquer des opérations mathématiques avec les fonctions statiques de cette
 * classe.<br>
 * Notez qu'une opération sur une matrice ne modifie pas le tableau originel, un
 * autre sera fabriqué.<br>
 * Utilitairement, pour créer un tableau 2D float, voir les méthodes
 * {@link Matrix.build} et {@link Matrix.apply}.
 * 
 */

public class Matrix {
	private Matrix() {
	}

	@FunctionalInterface
	public interface FloatUnaryFunction {
		public float apply(float i);
	}

	@FunctionalInterface
	public interface FloatBinaryFunction {
		public float apply(float i, float j);
	}

	// when you only have 8gb of ram :/
	@FunctionalInterface
	public interface FloatBinaryIntFunction {
		public float apply(int i, int j);
	}

	/**
	 * Fabrique une matrice {@code float[][]} de taille {@code h} x {@code w} avec
	 * une valeur retournée par le {@code block} à l'élément {@code matrix[i][j]}
	 * 
	 * @param h     - La taille d'une colonne.
	 * @param w     - La taille d'une ligne.
	 * @param block - Une opération lambda dans le style {@code (i,j)->(i+1 + j+1)}
	 *              qui indique la valeur de l'élément {@code matrix[i][j]}.
	 * @return Un tableau 2D {@code float[][]} représentant la matrice.
	 */
	public static float[][] build(int h, int w, FloatBinaryIntFunction block) {
		float[][] ary = new float[h][w];
		for (int i = 0; i < h; i++) {
			for (int j = 0; j < w; j++) {
				ary[i][j] = block.apply(i, j);// (float) block.applyAsDouble(i, j);
			}
		}
		
		return ary;
	}

	/**
	 * Retourne la copie d'une matrice {@code float[][]} existante<br>
	 * où l'élément matrix[i][j] est modifié par l'opération lambda {@code block}.
	 * <br>
	 * Ne modifie pas la matrice d'origine.
	 * 
	 * @param matrix
	 * @param block
	 * @return
	 */
	public static float[][] apply(float[][] matrix, FloatUnaryFunction block) {
		int h = matrix.length, w = matrix[0].length;
		float[][] ary = new float[h][w];
		for (int i = 0; i < h; i++) {
			for (int j = 0; j < w; j++) {
				ary[i][j] = block.apply(matrix[i][j]);
			}
		}
		return ary;
	}

	/**
	 * Combine deux matrices {@code float[][]} ensemble où l'élément[i][j]
	 * correspond à l'opérateur block avec pour arguments matrix1[i][j] et
	 * matrix2[i][j].
	 * 
	 * @param matrix1
	 * @param matrix2
	 * @param block   - une expression lambda dans le style
	 *                :{@code (m1e, m2e)->(m1e + m2e)}
	 * @return
	 */
	public static float[][] combine(float[][] matrix1, float[][] matrix2, FloatBinaryFunction block) {
		int h = matrix1.length, w = matrix1[0].length;
		if (h != matrix2.length || w != matrix2[0].length)
			throw new IllegalArgumentException("matrix1 dont match matrix2 size!");
		float[][] ary = new float[h][w];
		for (int i = 0; i < h; i++) {
			for (int j = 0; j < w; j++) {
				ary[i][j] = block.apply(matrix1[i][j], matrix2[i][j]);
			}
		}
		return ary;
	}

	/**
	 * Fabrique une matrice diagonale {@code float[][]} d'ordre {@code n} <br>
	 * dont tous les éléments de la diagonale principale sont de grandeur {@code k}.
	 * 
	 * @param n - La taille d'une colonne et d'une ligne du tableau.
	 * @param k - La valeur scalaire à affubler aux éléments de la diagonale
	 *          principale.
	 * @return Un tableau 2D {@code float[][]} représentant la matrice scalaire.
	 */
	public static float[][] loadScalar(int n, float k) {
		return Matrix.build(n, n, (i, j) -> i == j ? k : 0);
	}

	/**
	 * Fabrique une matrice identité {@code float[][]} d'ordre {@code n}.
	 * 
	 * @param n - La taille d'une colonne et d'une ligne du tableau.
	 * @return Un tableau 2D {@code float[][]} représentant la matrice identité.
	 */
	public static float[][] loadIdentity(int n) {
		return Matrix.loadScalar(n, 1);
	}

	/**
	 * Fabrique une matrice nulle {@code float[][]}.
	 * 
	 * @param w - La taille d'une d'une ligne du tableau.
	 * @param h - La taille d'une colonne du tableau.
	 * @return Un tableau 2D {@code float[][]} représentant la matrice nulle.
	 */
	public static float[][] loadZero(int h, int w) {
		return Matrix.build(w, h, (i, j) -> 0);
	}

	/**
	 * Fabrique une matrice de uns {@code float[][]}.
	 * 
	 * @param w - La taille d'une d'une ligne du tableau.
	 * @param h - La taille d'une colonne du tableau.
	 * @return Un tableau 2D {@code float[][]} représentant la matrice nulle.
	 */
	public static float[][] loadOnes(int h, int w) {
		return Matrix.build(h, w, (i, j) -> 1);
	}

	/**
	 * Fabrique une matrice ligne {@code float[][]} à partir d'un vecteur de taille
	 * n.
	 * 
	 * @param vector - un tableau 1D {@code float[]}.
	 * @return Un tableau 2D {@code float[][]} représentant la matrice ligne.
	 */
	public static float[][] loadRowVector(float[] vector) {
		int vw = vector.length;
		float[][] ary = new float[1][vw];
		for (int i = 0; i < vw; i++) {
			ary[0][i] = vector[i];
		}
		return ary;
	}

	/**
	 * Fabrique une matrice colonne {@code float[][]} à partir d'un vecteur de
	 * taille n.
	 * 
	 * @param vector - un tableau 1D {@code float[]}.
	 * @return Un tableau 2D {@code float[][]} représentant la matrice colonne.
	 */
	public static float[][] loadColumnVector(float[] vector) {
		int vw = vector.length;
		float[][] ary = new float[vw][1];
		for (int i = 0; i < vw; i++) {
			ary[i][0] = vector[i];
		}
		return ary;
	}

	/**
	 * Il s'agit de l'opération matricielle de multiplication entre 2 matrices.
	 * 
	 * @param matrix1 - La première matrice {@code float[][]}.
	 * @param matrix2 - La deuxième matrice {@code float[][]}.
	 * @return - La nouvelle matrice {@code float[][]}.
	 */
	public static float[][] multiply(float[][] matrix1, float[][] matrix2) {
		int w1 = matrix1[0].length;
		if (w1 != matrix2.length)
			throw new IllegalArgumentException(
					"matrix1 w1 (" + w1 + ") don't match matrix2 h2 (" + matrix2.length + ")!");

		int h1 = matrix1.length, w2 = matrix2[0].length;
		float[][] ary = new float[h1][w2];
		for (int i = 0; i < h1; i++) {
			for (int j = 0; j < w2; j++) {
				for (int k = 0; k < w1; k++) {

					ary[i][j] += matrix1[i][k] * matrix2[k][j];
					if (Float.isNaN(ary[i][j]))
						throw new IllegalArgumentException(matrix1[i][k] + " " + matrix2[k][j]);
				}
			}
		}

		// Matrix.clip(ary, 1.0f, 0.5f);
		return ary;
	}

	/**
	 * Addition arithmétique entre deux matrices.
	 * 
	 * @param m1
	 * @param m2
	 * @return
	 */
	public static float[][] addition(float[][] m1, float[][] m2) {
		int h = m1.length, w = m1[0].length;
		if (h != m2.length || w != m2[0].length)
			throw new IllegalArgumentException(
					"matrix1 (" + h + "," + w + ")dont match matrix2 " + m2.length + "," + m2[0].length + "size!");
		float[][] ary = new float[h][w];
		for (int j = 0; j < h; j++)
			for (int i = 0; i < w; i++)
				ary[j][i] = m1[j][i] + m2[j][i];
		return ary;
	}

	/**
	 * Addition arithmétique des éléments de m2 sur m1. Ne modifie que la première
	 * matrice.
	 * 
	 * @param m1
	 * @param m2
	 * @return
	 */
	public static void add(float[][] m1, float[][] m2, float rate) {
		int h = m1.length, w = m1[0].length;
		if (h != m2.length || w != m2[0].length)
			throw new IllegalArgumentException(
					"matrix1 (" + h + "," + w + ")dont match matrix2 " + m2.length + "," + m2[0].length + "size!");
		for (int j = 0; j < h; j++)
			for (int i = 0; i < w; i++)
				m1[j][i] += m2[j][i] * rate;
	}

	public static void substract(float[][] m1, float[][] m2, float rate) {
		int h = m1.length, w = m1[0].length;
		if (h != m2.length || w != m2[0].length)
			throw new IllegalArgumentException(
					"matrix1 (" + h + "," + w + ")dont match matrix2 " + m2.length + "," + m2[0].length + "size!");
		for (int j = 0; j < h; j++)
			for (int i = 0; i < w; i++)
				m1[j][i] -= m2[j][i] * rate;
	}

	/**
	 * Addition arithmétique entre trois matrices.
	 * 
	 * @param m1
	 * @param m2
	 * @return
	 */
	public static float[][] addition(float[][] m1, float[][] m2, float[][] m3) {
		int h = m1.length, w = m1[0].length;
		if (h != m2.length || w != m2[0].length || h != m3.length || w != m3[0].length)
			throw new IllegalArgumentException(
					"matrix1 (" + h + "," + w + ")dont match matrix2 " + m2.length + "," + m2[0].length + "size!");
		float[][] ary = new float[h][w];
		for (int j = 0; j < h; j++)
			for (int i = 0; i < w; i++)
				ary[j][i] = m1[j][i] + m2[j][i] + m3[j][i];
		return ary;
	}

	/**
	 * Soustraction arithmétique entre deux matrices.
	 * 
	 * @param matrix1
	 * @param matrix2
	 * @return
	 */
	public static float[][] substraction(float[][] m1, float[][] m2) {
		int h = m1.length, w = m1[0].length;
		float[][] ary = new float[h][w];
		for (int j = 0; j < h; j++)
			for (int i = 0; i < w; i++)
				ary[j][i] = m1[j][i] - m2[j][i];

		return ary;
	}

	/**
	 * Multiplication arithmétique entre chaque éléments des matrices.
	 * 
	 * @param matrix1
	 * @param matrix2
	 * @return
	 */
	public static float[][] multiplication(float[][] m1, float[][] m2) {
		/* Matrix.combine trop couteux */
		int h = m1.length, w = m1[0].length;
		float[][] ary = new float[h][w];
		for (int j = 0; j < h; j++)
			for (int i = 0; i < w; i++) {
				ary[j][i] = m1[j][i] * m2[j][i];
				if (Float.isNaN(ary[j][i]))
					throw new IllegalArgumentException(m1[j][i] + " " + m2[j][i]);
				if (Float.isInfinite(ary[j][i]))
					throw new IllegalArgumentException(m1[j][i] + " " + m2[j][i]);
			}
		// Matrix.clip(ary, 1.0f, 0.5f);
		return ary;
	}

	public static void clip(float[][] m, float d, float u) {
		int h = m.length, w = m[0].length;
		for (int j = 0; j < h; j++)
			for (int i = 0; i < w; i++)
				if (m[j][i] > u)
					m[j][i] = u;
				else if (m[j][i] < d)
					m[j][i] = d;

	}

	/**
	 * matrix vector substraction DONT USE
	 * 
	 * @param matrix
	 * @param vector
	 * @return
	 */
	public static float[][] substraction(float[][] matrix, float[] vector) {
		int w = vector.length, h = matrix.length;

		float[][] mat = new float[h][w];
		for (int j = 0; j < h; j++) {
			for (int i = 0; i < w; i++) {
				mat[j][i] = matrix[j][i] - vector[i];
			}
		}
		return mat;
	}
	
	public static void substraction_(float[][] matrix, float[] vector) {
		int w = vector.length, h = matrix.length;

		for (int j = 0; j < h; j++) {
			for (int i = 0; i < w; i++) {
				matrix[j][i] -= vector[i];
			}
		}
	}

	public static float[][] substraction(float[][] m1, float[][] m2, float[] vector) {
		int w = vector.length, h = m1.length;

		float[][] mat = new float[h][w];
		for (int j = 0; j < h; j++) {
			for (int i = 0; i < w; i++) {
				mat[j][i] = m1[j][i] - m2[j][i] - vector[i];
			}
		}
		return mat;
	}
	
	public static void substraction_(float[][] m1, float[][] m2, float[] vector) {
		int w = vector.length, h = m1.length;

		
		for (int j = 0; j < h; j++) {
			for (int i = 0; i < w; i++) {
				m1[j][i] = m1[j][i] - m2[j][i] - vector[i];
			}
		}
	}

	/**
	 * matrix vector addition DONT USE
	 * 
	 * @param matrix
	 * @param vector
	 * @return
	 */
	public static float[][] addition(float[][] matrix, float[] vector) {
		int w = matrix[0].length, h = matrix.length;

		float[][] mat = new float[h][w];
		for (int j = 0; j < h; j++) {
			for (int i = 0; i < w; i++) {
				mat[j][i] = matrix[j][i] + vector[i];
			}
		}
		return mat;
	}
	
	public static void addition_(float[][] matrix, float[] vector) {
		int w = matrix[0].length, h = matrix.length;

		
		for (int j = 0; j < h; j++) 
			for (int i = 0; i < w; i++) 
				matrix[j][i] += vector[i];
			
	}

	/**
	 * matrix vector division DONT USE
	 * 
	 * @param matrix
	 * @param vector
	 * @return
	 */
	public static float[][] division(float[][] matrix, float[] vector) {
		int w = vector.length, h = matrix.length;
		float[][] mat = new float[h][w];
		for (int j = 0; j < h; j++) {
			for (int i = 0; i < w; i++) {
				mat[j][i] = matrix[j][i] / vector[i];
			}
		}
		return mat;
	}

	public static float[][] division(float[][] m1, float[][] m2) {
		int w = m1[0].length, h = m1.length;
		float[][] mat = new float[h][w];
		for (int j = 0; j < h; j++) {
			for (int i = 0; i < w; i++) {
				mat[j][i] = m1[j][i] / m2[j][i];
			}
		}
		return mat;
	}

	/**
	 * matrix vector multiplication DONT USE
	 * 
	 * @param matrix
	 * @param vector
	 * @return
	 */
	public static float[][] multiplication(float[][] matrix, float[] vector) {
		int w = matrix[0].length, h = matrix.length;
		// System.out.println(vector.length + " " + h);
		float[][] mat = new float[h][w];
		for (int j = 0; j < h; j++) {
			for (int i = 0; i < w; i++) {
				mat[j][i] = matrix[j][i] * vector[i];
			}
		}
		return mat;
	}
	
	public static void multiplication_(float[][] matrix, float[] vector) {
		int w = matrix[0].length, h = matrix.length;
		
		for (int j = 0; j < h; j++) {
			for (int i = 0; i < w; i++) {
				matrix[j][i] *= vector[i];
			}
		}
	}

	public static float[][] multiplication(float[][] matrix, float[] vector1, float[] vector2) {
		int w = matrix[0].length, h = matrix.length;
		// System.out.println(vector.length + " " + h);
		float[][] mat = new float[h][w];
		for (int j = 0; j < h; j++) {
			for (int i = 0; i < w; i++) {
				mat[j][i] = matrix[j][i] * vector1[i] * vector2[i];
			}
		}
		return mat;
	}
	
	public static void multiplication_(float[][] matrix, float[] vector1, float[] vector2) {
		int w = matrix[0].length, h = matrix.length;
		
		for (int j = 0; j < h; j++) {
			for (int i = 0; i < w; i++) {
				matrix[j][i] *= vector1[i] * vector2[i];
			}
		}
	}

	public static float[][] multiplication(float[][] m1, float[][] m2, float[][] m3) {
		int h = m1.length, w = m1[0].length;
		float[][] ary = new float[h][w];
		for (int j = 0; j < h; j++)
			for (int i = 0; i < w; i++)
				ary[j][i] = m1[j][i] * m2[j][i] * m3[j][i];
		return ary;
	}

	public static float[][] multiplication(float[][] m1, float[] vector, float f) {
		int h = m1.length, w = m1[0].length;
		float[][] ary = new float[h][w];
		for (int j = 0; j < h; j++)
			for (int i = 0; i < w; i++)
				ary[j][i] = m1[j][i] * vector[i] * f;
		return ary;
	}

	public static float[][] multiplication(float[][] m1, float[][] m2, float f) {
		int h = m1.length, w = m1[0].length;
		float[][] ary = new float[h][w];
		for (int j = 0; j < h; j++)
			for (int i = 0; i < w; i++)
				ary[j][i] = m1[j][i] * f * m2[j][i];
		return ary;
	}

	public static float[][] multiplication(float[][] m1, float[][] m2, float[][] m3, float f) {
		int h = m1.length, w = m1[0].length;
		float[][] ary = new float[h][w];
		for (int j = 0; j < h; j++)
			for (int i = 0; i < w; i++)
				ary[j][i] = m1[j][i] * m2[j][i] * m3[j][i] * f;
		return ary;
	}

	public static int N(float[][] matrix) {
		return matrix.length * matrix[0].length;
	}

	public static float sum(float[][] matrix) {
		float sum = 0;
		for (int i = 0; i < matrix.length; i++)
			for (int j = 0; j < matrix[0].length; j++)
				sum += matrix[i][j];
		return sum;
	}

	public static float[] rowsum(float[][] matrix) {
		float[] ary = new float[matrix.length];

		for (int j = 0; j < matrix.length; j++) {
			for (int i = 0; i < matrix[0].length; i++) {
				ary[j] += matrix[j][i];
			}
		}
		return ary;
	}

	public static float[] rowmean(float[][] matrix) {
		float[] ary = new float[matrix.length];
		int w = matrix[0].length;
		for (int j = 0; j < matrix.length; j++) {
			for (int i = 0; i < w; i++) {
				ary[j] += matrix[j][i];
			}
			ary[j] /= w;
		}
		return ary;
	}

	public static float[] colsum(float[][] matrix) {
		float[] ary = new float[matrix[0].length];

		for (int i = 0; i < matrix[0].length; i++) {
			for (int j = 0; j < matrix.length; j++) {
				ary[i] += matrix[j][i];
			}
		}
		return ary;
	}

	public static float[] colmean(float[][] matrix) {
		float[] ary = new float[matrix[0].length];
		int w = matrix[0].length, h = matrix.length;
		for (int i = 0; i < w; i++) {
			for (int j = 0; j < h; j++) {
				ary[i] += matrix[j][i];
			}
			ary[i] /= h;
		}
		return ary;
	}

	/**
	 * faster than Matrix.apply(m,e->math.pow(e,2)). DO NOT USE
	 * 
	 * @param matrix
	 * @return
	 */
	public static float[][] square(float[][] matrix) {

		int w = matrix[0].length, h = matrix.length;
		float[][] ary = new float[h][w];
		for (int j = 0; j < h; j++)
			for (int i = 0; i < w; i++)
				ary[j][i] = (float) Math.pow(matrix[j][i], 2);
		return ary;
	}

	
	
	public static float[][] invsqrt(float[][] matrix, float e) {

		int w = matrix[0].length, h = matrix.length;
		float[][] ary = new float[h][w];
		for (int i = 0; i < w; i++)
			for (int j = 0; j < h; j++)
				ary[j][i] = (float) (1d / (Math.sqrt(matrix[j][i]) + e));
		return ary;
	}

	public static float invSqrt(float x) {
		float xhalf = 0.5f * x;
		int i = Float.floatToIntBits(x);
		i = 0x5f3759df - (i >> 1);
		x = Float.intBitsToFloat(i);
		x *= (1.5f - xhalf * x * x);
		return x;
	}

	public static float mean(float[][] matrix) {
		return Matrix.sum(matrix) / (float) (Matrix.N(matrix));
	}

	public static float[][] add(float[][] matrix, float scalar) {
		int h = matrix.length, w = matrix[0].length;
		float[][] ary = new float[h][w];
		for (int j = 0; j < h; j++)
			for (int i = 0; i < w; i++)
				ary[j][i] = matrix[j][i] + scalar;
		return ary;
	}

	public static float[][] substract(float[][] matrix, float scalar) {
		int h = matrix.length, w = matrix[0].length;
		float[][] ary = new float[h][w];
		for (int j = 0; j < h; j++)
			for (int i = 0; i < w; i++)
				ary[j][i] = matrix[j][i] - scalar;
		return ary;
	}

	/**
	 * Multiplication entre un scalaire {@code float} et une matrice
	 * {@code float[][]}.
	 * 
	 * @param matrix - La matrice {@code float[][]}.
	 * @param scalar - La valeur scalaire à multiplier à l'élément matrix[i][j].
	 * @return - La nouvelle matrice du résultat de cette opération
	 *         {@code float[][]}.
	 */
	public static float[][] multiply(float[][] matrix, double scalar) {
		int h = matrix.length, w = matrix[0].length;
		float[][] ary = new float[h][w];
		for (int j = 0; j < h; j++)
			for (int i = 0; i < w; i++)
				ary[j][i] = (float) (matrix[j][i] * scalar);
		return ary;
	}

	public static void multiply_(float[][] matrix, double scalar) {
		int h = matrix.length, w = matrix[0].length;
		for (int j = 0; j < h; j++)
			for (int i = 0; i < w; i++)
			 matrix[j][i] *= scalar;
	}
	
	/**
	 * Division entre un scalaire {@code float} et une matrice {@code float[][]}.
	 * 
	 * @param matrix - La matrice {@code float[][]}.
	 * @param scalar - La valeur scalaire à diviser à l'élément matrix[i][j].
	 * @return - La nouvelle matrice du résultat de cette opération
	 *         {@code float[][]}.
	 */
	public static float[][] divide(float[][] matrix, double scalar) {
		int h = matrix.length, w = matrix[0].length;
		float[][] ary = new float[h][w];
		for (int j = 0; j < h; j++)
			for (int i = 0; i < w; i++)
				ary[j][i] = (float) (matrix[j][i] / scalar);
		return ary;
	}

	public static float[][] transpose(float[][] matrix) {
		// est trop gourmant en ressources
		// return Matrix.build(matrix[0].length, matrix.length, (i, j) ->
		// matrix[(int)
		// j][(int) i]);
		int h = matrix.length, w = matrix[0].length;
		float[][] ary = new float[w][h];
		for (int i = 0; i < w; i++)
			for (int j = 0; j < h; j++)
				ary[i][j] = matrix[j][i];
		return ary;
	}

	public static float[][] t(float[][] matrix) {
		return Matrix.transpose(matrix);
	}

	public static float[][] loadTranslation(float x, float y, float z) {
		return new float[][] { { 1, 0, 0, 0 }, { 0, 1, 0, 0 }, { 0, 0, 1, 0 }, { x, y, z, 1 } };
	}

	/* ne marche pas, utiliser loadRotationX(loadRotationY(loadRotationZ(m))) */
	public static float[][] loadRotation(double angle, float x, float y, float z) {
		float c = (float) Math.cos(Math.toRadians(angle));
		float s = (float) Math.sin(Math.toRadians(angle));
		float C = (1 - c);
		return new float[][] { { x * x * C + c, x * y * C + z * s, x * z * C - y * s, 0 },
				{ x * y * C - z * s, y * y * C + c, y * z * C + x * s, 0 },
				{ x * z * C + y * s, y * z * C - x * s, z * z * C + c, 0 }, { 0, 0, 0, 1 } };
	}

	public static float[][] loadRotationX(double angle) {
		float c = (float) Math.cos(Math.toRadians(angle));
		float s = (float) Math.sin(Math.toRadians(angle));
		return new float[][] { { 1, 0, 0, 0 }, { 0, c, -s, 0 }, { 0, s, c, 0 }, { 0, 0, 0, 1 } };
	}

	public static float[][] loadRotationY(double angle) {
		float c = (float) Math.cos(Math.toRadians(angle));
		float s = (float) Math.sin(Math.toRadians(angle));
		return new float[][] { { c, 0, s, 0 }, { 0, 1, 0, 0 }, { -s, 0, c, 0 }, { 0, 0, 0, 1 } };
	}

	public static float[][] loadRotationZ(double angle) {
		float c = (float) Math.cos(Math.toRadians(angle));
		float s = (float) Math.sin(Math.toRadians(angle));
		return new float[][] { { c, -s, 0, 0 }, { s, c, 0, 0 }, { 0, 0, 1, 0 }, { 0, 0, 0, 1 } };
	}

	/**
	 * Offre une représentation en 1D d'un tableau {@code ary} en 2D 4x4 qui est
	 * facilement lisible pour OpenGL.
	 * 
	 * @param matrix - Un tableau 2D de float représentant une matrice 4x4.
	 * @return Le nouveau {@code FloatBuffer}.
	 */
	public static FloatBuffer toMat4(float[][] matrix) {
		FloatBuffer buf = BufferUtils.createFloatBuffer(16);
		for (float[] row : matrix)
			buf.put(row);
		buf.flip();
		return buf;
	}

	/**
	 * sépare une input matrix en matrices de plus petites taille. aka
	 * 'mini-batches'. à utiliser dans un dnn avec batch norm.
	 * 
	 * @param batchSize - taille des mini-batches. si taille % batchSize != 0, la
	 *                  dernière mini-batche aura une taille adaptée à taille %
	 *                  batchSize.
	 * @param mat       - un tableau 2d a convertir en tableau 3d.
	 * @return un tableau 3d
	 */
	public static float[][][] toBatch(int batchSize, float[][] mat) {
		
		if (batchSize > mat.length)
			throw new IllegalArgumentException("batchSize bigger than matrix length!");
		int batchN = (int) Math.ceil(mat.length / (double) batchSize);
		float[][][] ary = new float[batchN][][];
		if (batchSize >= mat.length)
			ary[0] = mat;
		else {

			for (int i = 0; i < batchN - 1; i++) {

				ary[i] = new float[batchSize][];
				for (int j = 0; j < batchSize; j++)
					ary[i][j] = mat[i * batchSize + j];
			}
			int remain = mat.length % batchSize == 0 ? batchSize : mat.length % batchSize;
			ary[batchN - 1] = new float[remain][];
			for (int j = 0; j < remain; j++)
				ary[batchN - 1][j] = mat[batchSize * (batchN - 1) + j];
		}
		return ary;
	}

	/**
	 * méthode utilitaire qui n'a pas sa place pour un sample de faces de cube (3d)
	 * dans un classe 2d
	 * 
	 * @param batchSize
	 * @param mat
	 * @return
	 */
	public static float[][][][][] toBatch(int batchSize, float[][][][] mat) {
		int l = mat.length;
		if (batchSize > l)
			throw new IllegalArgumentException("batchSize bigger than matrix length!");
		int batchN = (int) Math.ceil(l / (double) batchSize);
		float[][][][][] ary = new float[batchN][][][][];
		if (batchSize >= l)
			ary[0] = mat;
		else {
			for (int i = 0; i < batchN - 1; i++) {

				ary[i] = new float[batchSize][][][];
				for (int j = 0; j < batchSize; j++)
					ary[i][j] = mat[i * batchSize + j];
			}
			int remain = l % batchSize == 0 ? batchSize : l % batchSize;
			ary[batchN - 1] = new float[remain][][][];
			for (int j = 0; j < remain; j++)
				ary[batchN - 1][j] = mat[batchSize * (batchN - 1) + j];
		}
		return ary;
	}

	/**
	 * Affiche une représentation litérale d'une matrice (float[][]).
	 * <p>
	 * Ainsi, le code suivant :<br>
	 * {@code
	 * Matrix.print(Matrix.loadIdentity(3));}
	 * <p>
	 * Affichera dans la console :<br>
	 * {@code [ 1.0 0.0 0.0 ]}<br>
	 * {@code [ 0.0 1.0 0.0 ]}<br>
	 * {@code [ 0.0 0.0 1.0 ]}
	 * 
	 * @param matrix - Un tableau 2D de float représentant une matrice.
	 */
	public static void print(float[][] matrix) {
		int h = matrix.length, w = matrix[0].length;
		for (int i = 0; i < h; i++) {
			System.out.print("[ ");
			for (int j = 0; j < w; j++) {
				System.out.print(matrix[i][j] + " ");
			}
			System.out.println("]");
		}
	}

	public static String shape(float[][] matrix) {
		return Arrays.toString(new int[] { matrix.length, matrix[0].length });
	}

	

}
