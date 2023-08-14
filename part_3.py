# import modules
import torch
import matplotlib.pyplot as plt
import math

def dragon_curve_transform(points, iterations):
    # Iteratively apply the transformations to generate Dragon Curve
    # The main source is https://en.wikipedia.org/wiki/Dragon_curve in the construction section
    # Using the iterated function system with real numbers

    # Also used https://larryriddle.agnesscott.org/ifs/heighway/heighway.htm as a guide

    for _ in range(iterations):
        
        # f2 matrix, using 45 degree angle
        f1_matrix = torch.tensor([
        [math.cos(math.radians(45)), -math.sin(math.radians(45))],
        [math.sin(math.radians(45)), math.cos(math.radians(45))]
        ], dtype=torch.float32)
    
        # Apply rotation, scaling, and translation
        # parallelization for matrix multiplication
        f1_points = (1/math.sqrt(2)) * torch.mm(f1_matrix, points)

        # f2 matrix, using 135 degree angle which is equivalent to -45
        f2_matrix = torch.tensor([
        [math.cos(math.radians(135)), -math.sin(math.radians(135))],
        [math.sin(math.radians(135)), math.cos(math.radians(135))]
        ], dtype=torch.float32)
    
        # Apply rotation, scaling, and translation
        f2_points = (1/math.sqrt(2)) * torch.mm(f2_matrix, points) + torch.tensor([[1.0], [0.0]])
    

        # Update points for next iteration
        points = torch.cat([f1_points, f2_points], dim=1)
    return points

# Initial point
points = torch.tensor([[0.0, 1.0], [0.0, 0.0]])

# Number of iterations
iterations = 28
dragon_points = dragon_curve_transform(points, iterations)

plt.figure(figsize=(10, 10))
plt.scatter(dragon_points[0].numpy(), dragon_points[1].numpy(), s=0.5)
plt.axis("equal")
plt.axis("off")
plt.show()
