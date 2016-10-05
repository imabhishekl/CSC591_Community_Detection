
1. This program needs the following python dependencies:  
    
   a. Numpy
   b. Scipy
   c. Networkx

2. Running the program

   python community.py <filePath>

3. Sample Input

   python community.py path/graph_file density_threshold_value 

4. Sample Output

   Community Detected: set([X,Y,Z,......])
   Community Detected: set([P,Q,R,......])
   .
   .
   .

5. Example 1:

   Input: (this graph file is included in submission)
   python community.py example_graph 0.75

   Output:
   Community Detected: set([1, 2, 3, 4, 5, 6, 7])
   Community Detected: set([16, 14, 15])

6. Example 2:
   
   Input:
   python community.py amazon/amazon.graph.small 0.5

   Output:
   Community Detected: set([984, 985, 986, 988, 1805, 989])
   Community Detected: set([2443, 2287])
  

