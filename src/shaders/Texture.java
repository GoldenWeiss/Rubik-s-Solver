/**
 * 
 * https://github.com/SilverTiger/lwjgl3-tutorial/blob/master/src/silvertiger/tutorial/lwjgl/graphic/Texture.java
 */

package shaders;

import static org.lwjgl.opengl.GL11.GL_RGBA;
import static org.lwjgl.opengl.GL11.GL_TEXTURE_2D;
import static org.lwjgl.opengl.GL11.GL_UNSIGNED_BYTE;
import static org.lwjgl.opengl.GL11.glTexImage2D;
import static org.lwjgl.opengl.GL30.*;
import static org.lwjgl.stb.STBImage.stbi_load;

import java.nio.ByteBuffer;
import java.nio.IntBuffer;

import org.lwjgl.system.MemoryStack;


/**
 * Texture interface for glGenTextures() using STBLInterface
 * @author Greyvitch
 *
 */
public class Texture
{
	private int id;
	
	private Texture()
	{
		id = glGenTextures();
	}
	
	public int getId()
	{
		return id;
	}

	public void bind()
	{
		glActiveTexture(GL_TEXTURE0 + id);
		glBindTexture(GL_TEXTURE_2D, id);
	}

	public void unbind()
	{
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, 0);
	}

	private static Texture buildTexture(int w, int h, ByteBuffer bf)
	{
		Texture tex = new Texture();
		
		glActiveTexture(GL_TEXTURE0 + tex.getId());
		tex.bind();

		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA,
				GL_UNSIGNED_BYTE, bf);

		
		return tex;
	}

	public static Texture loadTexture(String path)
	{
		ByteBuffer bf;
		int width, height;
		try (MemoryStack stack = MemoryStack.stackPush())
		{
			IntBuffer w = stack.mallocInt(1);
			IntBuffer h = stack.mallocInt(1);
			IntBuffer comp = stack.mallocInt(1);

			bf = stbi_load(path, w, h, comp, 4);
			if (bf == null)
				throw new RuntimeException("Failed to load a texture file!");

			width = w.get();
			height = h.get();
		}
		return buildTexture(width, height, bf);
	}
}
