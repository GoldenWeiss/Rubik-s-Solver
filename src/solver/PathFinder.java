package solver;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;

import cube.Cube;
import cube.RNotation;

public class PathFinder
{

	int[][][][] goalState;
	float[] c3goalState;
	private PriorityQueue<QueueNode> open;

	private Map<QueueNode, Float> cost;
	private Map<QueueNode, QueueNode> visited;

	private NeuralNet nn;
	private List<RNotation> path;

	public PathFinder(int[][][][] pGoalState, NeuralNet pnn)
	{

		goalState = Cube.copy(pGoalState);
		c3goalState = Cube.toCube3(goalState);
		nn = pnn;
		path = new ArrayList<>();
		
	}

	public void start(int[][][][] pStart)
	{

		// early exit
		if (Cube.same(pStart, goalState))
			return;

		path.clear();
		QueueNode st = new QueueNode(0, pStart);

		cost = new HashMap<>();
		visited = new HashMap<>();
		open = new PriorityQueue<>(11, new Comparator<QueueNode>()  {
			@Override
			public int compare(QueueNode a, QueueNode b) {
				return Float.compare(a.getPriority(), b.getPriority());
			}
		});

		cost.put(st, 0f);
		visited.put(st, null);
		open.add(st);

		QueueNode current = null;

		System.out.println(hCost(Cube.toCube3(pStart)));
		System.out.println(hCost(st));
		while (!open.isEmpty())
		{
			current = open.poll();
			
			
			if (Arrays.equals(current.getC3data(), c3goalState)) {
				System.out.println("Reached goal state!");
				break;
			}
			for (QueueNode point : getNeightbours(current.getData()))
			{

				float new_cost = cost.get(current) + 1f;

				if (!cost.containsKey(point) || new_cost < cost.get(point))
				{
					cost.put(point, new_cost);
					point.setPriority(new_cost + hCost(point));
					open.add(point);
					visited.put(point, current);
				}
			}
		}

		System.out.println(open.isEmpty());
		if (!open.isEmpty())
		{
			// reconstruct path
			// current.getData() is equal to goalState
			do {
				
				path.add(current.getAction());
				current = visited.get(current);
			} while (current.getAction() != null);
			Collections.reverse(path);
		}

	}

	public List<RNotation> getPath()
	{
		return path;
	}

	private float hCost(float[] pc3data)
	{
		nn.setBatchSize(1);
		nn.setInputLayer(new float[][]
		{ pc3data });
		nn.propagation();
		return nn.getResultLayer()[0][0];
	}

	private float hCost(QueueNode point)
	{
		return hCost(point.getC3data());
	}

	public QueueNode[] getNeightbours(int[][][][] pCube)
	{
		QueueNode[] n = new QueueNode[12];
		for (int i = 0; i < 12; i++)
			n[i] = new QueueNode(0, Cube.rotate(pCube, RNotation.fromId(i)),
					RNotation.fromId(i));

		return n;
	}

	public class QueueNode
	{

		private float priority;
		private int[][][][] data;

		private float[] c3data;
		private int hc3data; // static
		private RNotation action;

		public QueueNode(float pPriority, int[][][][] pData, RNotation pAction)
		{
			priority = pPriority;
			data = pData;
			hc3data = Arrays.hashCode(c3data = Cube.toCube3(data));
			
			action = pAction;
		}

		public QueueNode(float pPriority, int[][][][] pData)
		{
			this(pPriority, pData, null);
		}


		public boolean equals(Object o)
		{
			return (o instanceof QueueNode)
					&& hc3data ==((QueueNode)o).hashcode();
		}

		public int hashcode()
		{
			return hc3data;
		}

		/**
		 * @return the priority
		 */
		public float getPriority()
		{
			return priority;
		}

		public void setPriority(float p)
		{
			priority = p;
		}

		public int[][][][] getData()
		{
			return data;
		}

		public float[] getC3data()
		{
			return c3data;
		}

		public RNotation getAction()
		{
			return action;
		}

	}
}
