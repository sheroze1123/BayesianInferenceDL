from dolfin import *
import matplotlib
matplotlib.use('macosx')
import matplotlib.pyplot as plt
import mshr

geometry = mshr.Rectangle(Point(0.5, 0.0), Point(0.75, 0.5)) \
        + mshr.Rectangle(Point(0.25, 0.0), Point(0.45, 0.5))

mesh = mshr.generate_mesh(geometry, 10)

plot(mesh)
plt.show()
