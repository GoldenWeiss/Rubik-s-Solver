package shapes;

public class P2

{
	private float[] coords;
	
	public P2() {
		this(0, 0);
	}

	public P2(float px, float py) {
		coords = new float[] {px, py};
	}
	
	public float[] getCoords() {
		return coords;
	}
	public void setX(float px) {
		coords[0] = px;
	}
	public void setY(float py) {
		coords[1] = py;
	}
	public float getX() {
		return coords[0];
	}
	public float getY() {
		return coords[1];
	}
}
