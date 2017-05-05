# HexacopterController
The controller for the attitudes of the Hexacopter
including 2 PID controller 
and command mixer from velocities feedback command to 6 rotor speeds.

## PID
PID is divided into 2 part, inner and outer controller.  
The inner controller is "PID_V" for the velocities(vertical speed and roll, pitch and yaw angle speeds)
outer controller is "PID_A" for attitude(altitude and roll, pitch and yaw angle).


### parameters
Gains can be set by setter method.
#### method name definition
-[ ] vertical speed  
        `set_gain_w`
-[ ] roll speed  
        `set_gain_p`
-[ ] pitch speed  
        `set_gain_q`
-[ ] yaw speed  
        `set_gain_r`
-[ ] vertical position  
        `set_gain_z`
-[ ] roll angle  
        `set_gain_phi`
-[ ] pitch angle  
        `set_gain_theta`
-[ ] yaw angle  
        `set_gain_psi`

to achieve this controller architecture
-[x] multi dimensional PID unit

## CommandMixer
The command mixer is a matrix that is psudo-invert matrix of the matrix E of equation of motion of the hexacopter.
