# The percepts for the farmer agent are as follows:
# farmer_location, fox_location, chicken_location, grain_location

# The actions are move(animal=none), if an animal is specified it will be moved to the opposite side along with the farmer,
# If no animal is specified the farmer will move alone.


# The actions and precept sequence needed to solve the problem are as follows
# [Farmer=A, Fox=A, Chicken=A, Grain=A] -> move(chicken)
# [Farmer=B, Fox=A, Chicken=B, Grain=A] -> move()
# [Farmer=A, Fox=A, Chicken=B, Grain=A] -> move(fox)
# [Farmer=B, Fox=B, Chicken=B, Grain=A] -> move(chicken)
# [Farmer=A, Fox=B, Chicken=A, Grain=A] -> move(grain)
# [Farmer=B, Fox=B, Chicken=A, Grain=B] -> move()
# [Farmer=A, Fox=B, Chicken=A, Grain=B] -> move(chicken)
# [Farmer=B, Fox=B, Chicken=B, Grain=B] -> goal state
