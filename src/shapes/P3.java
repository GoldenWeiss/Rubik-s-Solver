package shapes;

public class P3

{

	public float x, y, z;

	private float[] coords;
	
	public P3() {
		this(0, 0, 0);
	}

	public P3(float px, float py, float pz) {
		x = px;
		y = py;
		z = pz;
		
		coords = new float[] {px, py, pz};
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
	public void setZ(float pz) {
		coords[2] = pz;
	}
	public float getX() {
		return coords[0];
	}
	public float getY() {
		return coords[1];
	}
	public float getZ() {
		return coords[2];
	}
	
	public void rotateX(int direction) {
		float tz = getZ();
		setZ(direction*getY());
		setY(-direction*tz);
	}
	public void rotateY(int direction) {
		float tz = getZ();
		setZ(-direction*getX());
		setX(direction*tz);
	}
	public void rotateZ(int direction) {
		float tx = getX();
		setX(-direction*getY());
		setY(direction*tx);
	}
}
