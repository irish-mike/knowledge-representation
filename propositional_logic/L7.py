# Bob has a state-of-the-art smart home system. The sys-
# tem has various rules set up to manage the appliances and
# features of the house eﬀiciently. Bob wants to ensure that
# when he gets home from work, his house is comfortable
# and welcoming.
from aima.logic import PropKB
from aima.utils import Expr

# if it’s dark outside, the living room lights should be turned on.

D = Expr('D') # It is Dark
L = Expr('L') # Lights are on
C = Expr('C') # it is cold
H = Expr('H') # heater is on
L18 = Expr('L18') # less than 18 degrees
A6 = Expr('A6') # After 6pm

KB = PropKB()

KB.tell(Expr('==>', D, L)) # if its dark the lights are on
KB.tell( Expr('==>', (L & C), H)) # Lights on, cold implies heater
KB.tell( Expr('==>', L18, C)) # Considered cold if it's less than 18 degrees
KB.tell( Expr('==>', A6, D)) # after 6pm implies its dark outside

# Scenario 1, Scenario: Bob enters his home at 7 pm. The temperature
# outside is 17°C. (EXPECT TRUE

KB.tell(A6) # its after 6pm (7pm)
KB.tell(L18) # its less than 18C (17)

heater_on = KB.ask_if_true(H)
print("Scenario 1, Scenario: Bob enters his home at 7 pm. The temperature outside is 17°C, Heater:", heater_on)

KB.retract(L18)
KB.tell(~L18) # its NOT less than 18C (18)
heater_on = KB.ask_if_true(H)
print("Scenario 2, The temperature outside is above 18°C, Heater:", heater_on)