# SED500 Assignment 1 Questions

## 1. You may notice that the cost function is based on Euler's method. What other method(s) might be applied to enable it to run more efficiently?

---

The idea of Euler's Method is bascially adding the ***step-size * slope*** to the previous value to estimate the next value.
However, in this problem, the slope of the voltage equation is unknown to us.  But we could calculate the slope of the error, the estimate the next step-size so that the error will be 0.
This assumes that the change in current is porpotional to the change in voltage.
Although this is not yet tested on complex circuits where the relationship between current and voltage might not be linear, it dramatically reduces the number of iterations that the algorithm takes to estimate the current for this simple circuit.
The original function takes around 60 loops in average and this one only takes 3 loops.

```cpp
void AnalogCircuit::CostFunctionV(double &current, double voltage) {
	double I1 = current;
    double J1 = 0.0;
	double J0 = 0.0;
	double alpha = 1.0;

	do {

		double sumVoltage = 0.0;
		list<Component*>::iterator it;
		for (it = component.begin(); it != component.end(); ++it) {
			double Vcomponent = (*it)->GetVoltage(I1, T);
			sumVoltage += Vcomponent;
		}
		J1 = sumVoltage - voltage;

        // Find the best step size
        // J1: J1-0, the different between error and ideal error (0)
        // alpha / (J1 - J0): amp per volt, the change in current needed to cause one unit of voltage change based on the current alpha 
		if (J1 != J0)
			alpha = (J1 * alpha) / (J1 - J0); 

		if (abs(J1) > tolerance) {
			if (J1 < 0) {//increase the current
				I1 += alpha;
			}
			else {
				I1 -= alpha;
			}
			J0 = J1;
		}

	} while (abs(J1) > tolerance);

    // file output...
	current = I1;
}

```

### **However, this algorithm does NOT work on non-linear circuits.**

The below code implements bi-section search, which deviates from the Euler's method but still implements the similar idea of taking steps. The step-size in this algorithm is just calculated differently. For the RLC circuit, this code takes around 30 iterations to find the current, which is around half of the original cost function.

In addition, this code works for non-linear circuits as well and has been tested on the diode resistor serie circuit. The result indicates that the performance has increased by 60%. 

```cpp
void AnalogCircuit::CostFunction(double &current, double voltage) {
	double I1 = current, J1 = 0.0,  J0 = 0.0;
	double min = -0.1, max = 0.1;
	double count = 0;

	auto sumVoltage = [&](double c, double t) {
		double sumVoltage = 0.0;
		list<Component*>::iterator it;
		for (it = component.begin(); it != component.end(); ++it) {
			double Vcomponent = (*it)->GetVoltage(c, t);
			sumVoltage += Vcomponent;
		}
		return sumVoltage;
	};
    // validate min, if min still causes J1 to be positive, double it and max = min
	do {
		J1 = sumVoltage(min, T) - voltage;
		if (J1 > 0) {
			max = min;
			min *= 2.0;
		}
	} while (J1 > 0);

    // validate max, if max still causes J1 to be negative, double it and min = max
	do {
		J1 = sumVoltage(max, T) - voltage;
		if (J1 < 0) {
			min = max;
			max *= 2.0;
		}
	} while (J1 < 0);

	do {
		J1 = sumVoltage(I1,T) - voltage;
		// if J1 is positive, current too large
		if (J1 > 0 && J1 > tolerance) {
			max = I1;
			I1 = (I1 + min) / 2.0;
		}

		if (J1 < 0 && -J1 > tolerance) {
			min = I1;
			I1 = (I1 + max) / 2.0;
		}

		// For diode, add the next line
		// if (I1 < -Is) I1 = -Is; //corner-case, the reverse saturation current of the diode
			

	} while (abs(J1) > tolerance);
	

	//file output...
	current = I1;
}
```





## 2. You may notice that the cost function has no intelligence. It does not learn. This heuristic algorithm does not comply with the principles of machine learning. How might you add intelligence to our cost function to enable it to run more efficiently?
---

In the previous question, we have a couple lines of code to calculate the step-size.

```cpp
if (J1 != J0)
	alpha = (J1 * alpha) / (J1 - J0); 
```

This code has given the cost function some sort of intelligence. It "learns" from the error and adjust its step size. **But it doesn't work for non-linear components.**

Although it's used in various numerical methods, in machine learning, adaptive learning rate is also commonly used.

In my opinion, it's some kind of learning because it doesn't adjust the step size by a fixed amount or rate, it 'knows' how to converge to the solution.

The code below is the implementation for the diode-resistor circuit.

The idea is as follow:

1. Have two different initial guess of the current **(I1, I2)** and the corresponding voltage values (V1, V2) and the errors **(J1, J2)**.

	Be careful choosing the initial values since if they are too large, the next estimated current value will be negative and smaller than **-Is** (in this diode example), then the voltage will be ***-inf***. The algorithm will not work.

2. Calculate the slope, $d= \dfrac{\Delta voltage}{\Delta current}$, which is essentially $\dfrac {J1-J2} {I1-I2}$.

3. Calculate the new x (current) where the line of the slope intercept horizontal line of the true voltage value (y) by calculating the step size using the slope.

	The step size alpha can be calculated using $\alpha = \dfrac{J2}{d}$.

4. Store **I2** and **J2** in **I1** and **I2** then calculate the new I2 by using **I1 - alpha** and calculate **J2** using the new **I2**.

5. Check if stopping criteria is met, if not, repeat steps 2-5.


Although this is more like pure math, it gives the algorithm information what the next step size should be.

Performance wise, it takes a maximum of 10 and a minimum of 3 iterations. 

In addition, the original costfunction and the bi-section search method get stuck not long after reverse bias starts. This algorithm somehow is able to find the solution by only using 4 iterations.

<img src="https://raw.githubusercontent.com/daysond/Diode/master/Diode/Diode/Capture.PNG?token=GHSAT0AAAAAACHH2QZQCCDB6H77FMWS4QX6ZJZ5GCQ" alt="drawing" width="600"/>

```cpp
void AACostFunctionV(Component* component[], const int NUM, double& current, const double Is, const double voltage) {
	// AA stands for Adaptive Alpha
	double I1 = 0.00000001;
	double I2 = 0.00000002;
	double J1 = 0.0;
	double J2 = 0.0;
	double alpha =0.005;
	const double tolerance = 0.0001;
	double d = 0; // slope 

	auto sumVoltage = [&](double c) {
		double sumVoltage = 0.0;
		for (int i = 0; i < NUM; ++i) {
			double Vcomponent = component[i]->GetVoltage(c);
			sumVoltage += Vcomponent;
		}
		return sumVoltage;
	};

	J1 = sumVoltage(I1);
	J2 = sumVoltage(I2);

	do {
	
		d = (J1 - J2) / (I1 - I2);
		alpha = J2 / d;

		I1 = I2;
		J1 = J2;
		I2 = I1 - alpha;
		if (I2 < -Is)
			I2 = -Is;
		J2 = sumVoltage(I2) - voltage;
	
	} while (abs(J2) > tolerance);


	current = I2;
}

```



## 3. Devise a completely different way to implement the cost function. Is there some quicker way of guessing the present current knowing the previous voltage and current?
---
## Basic Idea:
The essense of this problem is to find a current value that minimize the difference between the estimated sum of voltage and the true voltage value.
It is exactly an optimization problem, in which the error is optimized by searching for the global minimum.

One of the metaheuristic algorithms for optimization is PSO (Partical Swarm Optimization).

The PSO algorithm is one of the nature inspired optimization algorithm. It mimics the behaviour of a shool of fish, or a flock of birds.
For instance, while a bird flying and searching randomly for food, all birds in the flock can share their discovery and help the entire flock get the best hunt.
Or, a mumuration of starlings trying to mimimize engery use in traveling through space by observing the directions and velocities of other starlings around.

The idea is that, we put a particle swarm into the search space, give them behaviours which depends on the fitness function. The particals can exchange intel and will move towards the best location so far in a ceratin velocity while keep discovering better options on the way. 

This image shows the major steps of the PSO algortihm:

![img](https://www.researchgate.net/publication/329007429/figure/fig3/AS:693775361929219@1542420344803/Basic-structure-of-the-particle-swarm-optimization-PSO-algorithm.png)



## Algorithm Explaination:

First of all, the problem can be modeled as: given x, find f(x) such that the difference between the sum of voltage of different components and the true voltage value is minimized.

**f(x)** will be the fitness function that returns the error.

In the **PSO()** function code, we initialize the PSO parameters: 
    
- N: number of particles
- d: dimensions
- genrateion: max number of iteration
- limit: the boundary of the search space, the minimum and maximum current value.
- vlimit: velocity limit, maximum and minimum velocity
- w: particle inertia, how much should the particle move towards the best potision so far
- c1 & c2: particle and swarm learning rate

Then it will run the algorithm by putting N particles into the search space randomly and initialize their velocities.

The best positions (x and y) of both individual particals (local) and the swarm (global) is updated.

Next, the particle will start moving towards the best local so far (glocal optimum) while they discover if there's any better option along the way.

If a better global optimum or local optimum is found, the values will be updated.

The algorithm keeps running until it runs out of iteration or criteria is met.

If criteria is not met and the iteration has run out, the algorithm will run again and usually it will be able to find the optimum in the second run. 

## Futher optimization:

The PSO parameters are tweaked to ensure better performance. However, there are some isssues and room for improvement.

1. Search space: notice the search space is hardcoded. What if the circuit drains current that exceeds 0.4A? In this case, we could implement the code that's in the bi-section search algortihm to find out the boundry innitially. Since the V-peak is known, we could store the current value at V-peak and used them as the boundary later on.

2. The initial positions of the particles are random. However, given the previous current, we could drop particles around the previous current value so that the particles will be closer to the global optimum initially.

3. The velocity and direction can be optimized based on rate of change in current over change in voltage. 

4. Termination might be required to avoid infinite loops. i.e. it should take less than 10 attempts to estimate the current. Or else, it indicates some tweaks in parameters.

## Summary:

Overall, the PSO algorithm is useful for this particular problem even when non-linear components are added since it treats the fitness function as a blackbox. It's problem independent but can be modified to suit the problem better for optimized performance.

## PSO Implementation in Python (for RLC)


```python
def RVoltage(i):
    return i * r

def LVoltage(i):
    return l * (i - ip) / ts

def CVoltage(i):
    return vp + i * ts / c

def f(x):
    # Modeling the problem to minimize the tolerance
    # The problem can be modeled as f(x) = abs(sum_of_component_voltage - true_voltage)
    # Goal: minimize f(x)
    return abs(RVoltage(x) + LVoltage(x) + CVoltage(x) - C) # C: true voltage, a constant.


def PSO():
    # PSO parameters
    N = 35  # number of particle
    d = 1   # dimension, number of variables 
 
    generation = 40         # number of iterations
    limit = [-0.4, 0.4]     # x bound, search space.
    vlimit = [-0.05, 0.05]  # velocity limit
 
    w = 0.268       # [inertia weight] of particle 
                    # lower the w, faster the particle will go towards the global best, shall decrease as iteration increase
    c1 = 0.9        # [Cognitive constant] particle learning rate  
    c2 = 0.8        # [social constant] swarm learning rate     
    
    restart = True
    while restart:
        # Initialize the population
        x = limit[0] + (limit[1] - limit[0]) * np.random.rand(N, d) # random particle position
        v = np.random.rand(N, d)        # velocity
        xm = x.copy()                   # best particle position (x)
        ym = np.zeros(d)                # best swarm position (x), init with an array of [0]
        fxm = np.full(N, np.inf)        # best particle position (y) 
        fym = np.inf                    # best swarm position (y)


        iter = 1                        # PSO iteration count

        # Stop as soon as a minima less than tolerance is found or run out of iteration
        while f(ym[0]) >= tolerance and iter <=generation:
            fx = f(x)                   # current particle position (y)
            for i in range(N):
                if fx[i] < fxm[i]:
                    fxm[i] = fx[i]      # updating best particle position (y)
                    xm[i, :] = x[i, :]  # updatomg best particle position (x)
            if min(fxm) < fym:
                fym = min(fxm)          # updating best swarm position (y)
                nmin = np.argmin(fxm)
                ym = xm[nmin, :]        # updating best swarm position (x)
            
            # updating velocity
            v = w * v + c1 * np.random.rand() * (xm - x) + c2 * np.random.rand() * (np.tile(ym, (N, 1)) - x)
            # velocity limit
            v[v > vlimit[1]] = vlimit[1]
            v[v < vlimit[0]] = vlimit[0]
            # updating position x
            x = x + v
            # position limit
            x[x > limit[1]] = limit[1]
            x[x < limit[0]] = limit[0]
        
            iter += 1
        
        # Stop if a desired minima is found, otherwise re-run algorithm
        if f(ym[0]) < tolerance:
            restart = False

    return (ym[0], iter)

```


## 4. You may not like the way this project was designed and implemented. What changes would you suggest and why (ie. graphics, circuit components, using C++, ...).
---

I would modify the architecture/structure of the project. 

The project is not very modularized. For instance, functions related to the openGL library are scatters in **main** and the **AnalogCircuit** class. What I would do is create a class, say **Renderer**, for all the openGL related functions. In the **Renderer** class, we could have methods such as **SetupScreen(), DrawCoordinate(), StartRendering()** etc. 

The **AnalogCircuit** class kind of violates the '***S***' in the ***SOLID*** principle. Inside the **run()** method, it calculates the current using the voltage and renders the coordinates as well. And in the **CostFunction()**, it also display the voltage point for each component. For interaction with openGL, we could introduce an interface, say **CircuitVisualizer** between the **AnalogCircuit** Component and the **Renderer** Component and implement the oberver pattern. When the circuit starts, **CircuitVisualizer** will draw the axises, tickers and the legends. Then, when the circuit is running, it will draw the voltage values of each component. The **CircuitVisualizer** will be observing the **AnalogCircuit** and react to it.

Even though this approach introduces overhead, it modularizes the components and makes them reusable and scalable, allowing easy maintainence.

Further more, we could use strategy patterns or something similar to implement a set cost functions for different circuit configuration (simple linear, complex non-linear etc) as for linear components only, linear cost function will perform better than a general algorithm.

There are other suggestions as well such as allowing user to set the circuit parameters, auto horizontal scrolling effect when the voltages are rendering so it will not be limited to a fixed duration. 
