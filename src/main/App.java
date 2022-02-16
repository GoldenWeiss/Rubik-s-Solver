package main;
import shaders.*;
import solver.CNeuralInterface;
import solver.NeuralInterface;


import org.lwjgl.glfw.*;
import org.lwjgl.opengl.GL;
import org.lwjgl.system.MemoryStack;

import cube.*;
import math.Matrix;

import static org.lwjgl.opengl.GL11.*;
import static org.lwjgl.opengl.GL20.GL_FRAGMENT_SHADER;
import static org.lwjgl.opengl.GL20.GL_VERTEX_SHADER;
import static org.lwjgl.opengl.GL20.glUniformMatrix4fv;
import static org.lwjgl.opengl.GL30.*;
import static org.lwjgl.glfw.Callbacks.*;
import static org.lwjgl.glfw.GLFW.*;
import static org.lwjgl.system.MemoryUtil.*;
import static org.lwjgl.system.MemoryStack.*;


import java.nio.FloatBuffer;
import java.nio.IntBuffer;

public class App {
	
	public static float[][] vmatrix = Matrix.loadIdentity(4);
	public static FloatBuffer vm4 = Matrix.toMat4(vmatrix);
	
	private long wptr;
	private int WINDOW_WIDTH = 100;
	private int WINDOW_HEIGHT = 100;
	private String WINDOW_TITLE = "b i f r o s t <,-,>";
	
	private VertexBuffer buf;
	private GLSLProgram program;
	
	private SpriteCube sc;
	private CNeuralInterface ni;
	
	public void run() 
	{
		init();
		loop();
		stop();
	}
	private void init() 
	{
		
		if (!glfwInit())
			throw new IllegalStateException("Unable to initialize GLFW.");
		wptr = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, WINDOW_TITLE, NULL, NULL);
		if (wptr == NULL)
			throw new RuntimeException("Failed to create the GLFW window.");
		
		GLFWVidMode vidmode = glfwGetVideoMode(glfwGetPrimaryMonitor());
		glfwSetWindowPos(wptr, (vidmode.width() - WINDOW_WIDTH)/2, (vidmode.height() - WINDOW_HEIGHT)/2);
		glfwMakeContextCurrent(wptr);
		glfwSwapInterval(1);
		glfwShowWindow(wptr);
	}
	private void loadGLSLProgram() {
		/*
		float vertices[] = { 
				0, 0, 0, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f, 
				0, 0, 0, 1.0f, 1.0f, 1.0f, 0.0f, 1.0f, 
				0, 0, 0, 1.0f, 0.0f, 1.0f, 1.0f, 1.0f, 
				0, 0, 0, 1.0f, 1.0f, 1.0f, 1.0f, 0.0f }; cool ray effect */
		float vertices[] = { 
				0, 0, 0, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 
				0, 0, 0, 1.0f, 1.0f, 1.0f, 0.0f, 1.0f, 
				0, 0, 0, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 
				0, 0, 0, 1.0f, 1.0f, 1.0f, 1.0f, 0.0f }; 
		buf = new VertexBuffer();
		buf.setData(vertices);
		
		program = new GLSLProgram();
		int vert = GLSLProgram.buildShaderFromPath(GL_VERTEX_SHADER, "shaders/vertexShader.glsl");
		int frag = GLSLProgram.buildShaderFromPath(GL_FRAGMENT_SHADER, "shaders/fragmentShader.glsl");
		program.attachShader(vert);
		program.attachShader(frag);
		program.use();

		program.vertexAttribLocation("position", 3, GL_FLOAT, false, 8 * 4, 0);
		program.vertexAttribLocation("color", 3, GL_FLOAT, false, 8 * 4, 3 * 4);
		program.vertexAttribLocation("texcoord", 2, GL_FLOAT, false, 8 * 4, 6 * 4);
	}
	private void update() {
		// Resize sprites
		try(MemoryStack stack = stackPush()) {
			IntBuffer w = stack.mallocInt(1);
			IntBuffer h = stack.mallocInt(1);
			glfwGetWindowSize(wptr, w, h);
			glViewport(0,0, w.get(0), h.get(0));
		}
		glfwSetWindowTitle(wptr, WINDOW_TITLE + " " + ni.n);
	}
	
	private void draw() {
		sc.update();
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		
		sc.draw();
		glfwSwapBuffers(wptr);
	}
	private void loop() 
	{
		GL.createCapabilities();
		glEnable(GL_DEPTH_TEST);
		glClearColor(0.0f, 0.1f, 0.1f, 1.0f);
		loadGLSLProgram();
		
		Texture[] texs = new Texture[7];
		texs[0] = null;
		for (int i = 1; i < 7; i++) 
			texs[i] = Texture.loadTexture("textures/texture" + i + ".png");
		sc = new SpriteCube(program, buf, texs, Cube.loadSolved());
		
		// Actual main code
		ni = new CNeuralInterface(sc);
		Thread t = new Thread(ni);
		t.start();
		
		

		glfwSetWindowTitle(wptr, WINDOW_TITLE );
		while(!glfwWindowShouldClose(wptr)) {
			update();
			draw();
			glfwPollEvents(); // update events
		}
	}
	
	private void stop() 
	{
		glfwFreeCallbacks(wptr);
		glfwDestroyWindow(wptr);
		glfwTerminate();
	}
	public static void main(String[] args) 
	{
		
		App x = new App();
		x.run();
	}
}
