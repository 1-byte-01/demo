# =============================================================================
#  Task 2B  -  Balance the bot and turn it to each yaw setpoint
#  CoppeliaSim child script  -  BOILERPLATE
# =============================================================================
#
#  WHAT YOU HAVE TO DO
#  -------------------
#  Two things, at the same time:
#     1. keep the bot UPRIGHT                  ->  tilt stays near 0
#     2. turn the bot to the current SETPOINT  ->  yaw follows yaw_setpoint
#
#  The evaluator sends the setpoint as a float signal called "yaw_setpoint",
#  in RADIANS, measured from where the bot started:
#        0  ->  1.5708  ->  3.1416  ->  4.7124  ->  3.1416
#       (0     +90        +90        +90        -90 degrees)
#  Reading it is already done for you in sysCall_sensing().
#
#  Work through STEP 1..4 in order. Write your code only where it says
#
#        ####### ADD YOUR CODE HERE ######
#
#  Do not rename, add or move anything in the scene.
# =============================================================================

import math

###### GLOBAL VARIABLES ######

# --- Scene objects (filled in by sysCall_init) -----------------------------
reference_frame = None
front_motor = None  # drives the omni wheel: your ONLY input
drive_motor = None  # rear wheel: leave it alone, do not drive it

# --- The four states, updated every step in sysCall_sensing ---------------
tilt = 0.0  # lean of the bot, 0 when upright            [rad]
tilt_dot = 0.0  # how fast it leans                          [rad/s]
yaw = 0.0  # heading of the bot, 0 at the start         [rad]
yaw_dot = 0.0  # how fast it turns                          [rad/s]

# --- The setpoint, updated every step in sysCall_sensing ------------------
yaw_setpoint = 0.0  # where the evaluator wants the bot to face  [rad]

# --- Control ---------------------------------------------------------------
U = 0.0  # command sent to front_motor
dt = 0.01  # length of one simulation step              [s]

# Use the motor sign that worked in Task 2A (the Tuner printed it).
MOTOR_SIGN = -1

# The largest yaw offset your controller is allowed to see [rad].
# Too small: slow to turn. Too big: the bot falls on a 90 degree jump.
YAW_LIMIT = 0.35

# ---------------------------------------------------------------------------
#  YOUR GAINS FROM TASK 2A. Fill in ONE of the two blocks.
# ---------------------------------------------------------------------------

# Option A: LQR. The K the Task 2A LQR Tuner printed.
#     K = [ k_tilt_dot , k_tilt , k_yaw_dot , k_yaw ]
K = [0.0, 0.0, 0.0, 0.0]

# Option B: PID. The six gains from the Task 2A PID Tuner.
Kp_tilt, Ki_tilt, Kd_tilt = 140.100, 0.000, 30.973
Kp_yaw, Ki_yaw, Kd_yaw = 95.600, 0.000, 70.796

tilt_integral = 0.0
yaw_integral = 0.0


# You can add your own variables here.

##############################


# =============================================================================
#  Helpers, already written for you
# =============================================================================

def wrap_to_pi(angle):
    """Fold any angle into -pi .. +pi, so the bot always turns the short way."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def clamp(value, limit):
    """Keep value inside -limit .. +limit."""
    return max(-limit, min(limit, value))


def bike_tilt_yaw():
    """Tilt and yaw of the bot in radians, measured the same way as the
    Task 2A Tuners and the Task 2B evaluator."""
    m = sim.getObjectMatrix(drive_motor, reference_frame)
    axle = [m[2], m[6], m[10]]

    front = sim.getObjectPosition(front_motor, reference_frame)
    rear = sim.getObjectPosition(drive_motor, reference_frame)
    forward = [front[i] - rear[i] for i in range(3)]
    length = math.sqrt(sum(v * v for v in forward))
    forward = [v / length for v in forward]

    up_z = axle[0] * forward[1] - axle[1] * forward[0]

    t = math.atan2(-axle[2], up_z)
    y = math.atan2(-forward[0], forward[1])
    return t, y


# =============================================================================
def sysCall_init():
    # Runs ONCE, when the simulation starts.
    global sim, reference_frame, front_motor, drive_motor, dt
    global tilt, yaw, tilt_dot, yaw_dot
    global tilt_integral, yaw_integral, previous_yaw_setpoint

    sim = require('sim')

    # STEP 1: get the scene objects. Done for you.
    reference_frame = sim.getObject('/reference_frame')
    front_motor = sim.getObject('/reference_frame/spherical_joint/bike_respondable/front_motor')
    drive_motor = sim.getObject('/reference_frame/spherical_joint/bike_respondable/drive_motor')

    dt = sim.getSimulationTimeStep()

    # First reading, so the first tilt_dot and yaw_dot are not a jump
    tilt, yaw = bike_tilt_yaw()
    tilt_dot = 0.0
    yaw_dot = 0.0
    tilt_integral = 0.0
    yaw_integral = 0.0
    previous_yaw_setpoint = None

    ####### ADD YOUR CODE HERE ######
    # Anything else you want to set up once.

    #################################


# =============================================================================
def sysCall_sensing():
    # Runs every simulation step. Read the world here.
    global tilt, tilt_dot, yaw, yaw_dot, yaw_setpoint

    # The setpoint. Done for you. It stays 0 until the evaluator sends one.
    value = sim.getFloatSignal('yaw_setpoint')
    if value is not None:
        yaw_setpoint = value

    new_tilt, new_yaw = bike_tilt_yaw()

    tilt_dot = (new_tilt - tilt) / dt
    yaw_dot = wrap_to_pi(new_yaw - yaw) / dt

    tilt = new_tilt
    yaw = new_yaw

    ####### ADD YOUR CODE HERE ######
    #
    #  STEP 2: update the four states.
    #
    #    new_tilt, new_yaw = bike_tilt_yaw()
    #
    #    tilt_dot = (new_tilt - tilt) / dt
    #    yaw_dot  = wrap_to_pi(new_yaw - yaw) / dt
    #
    #    tilt = new_tilt
    #    yaw  = new_yaw

    #################################


# =============================================================================
def sysCall_actuation():
    # Runs every simulation step. Turn the states into one motor command.
    global U, tilt_integral, yaw_integral, previous_yaw_setpoint

    if previous_yaw_setpoint is None or yaw_setpoint != previous_yaw_setpoint:
        yaw_integral = 0.0
        previous_yaw_setpoint = yaw_setpoint

    yaw_offset = wrap_to_pi(yaw - yaw_setpoint)
    yaw_offset = clamp(yaw_offset, YAW_LIMIT)

    tilt_integral += tilt * dt
    yaw_integral += yaw_offset * dt

    U = (Kp_tilt * tilt + Ki_tilt * tilt_integral + Kd_tilt * tilt_dot) \
        - (Kp_yaw * yaw_offset + Ki_yaw * yaw_integral + Kd_yaw * yaw_dot)

    sim.setJointTargetVelocity(front_motor, -MOTOR_SIGN * U)

    ####### ADD YOUR CODE HERE ######
    #
    #  STEP 3: how far the bot is from the setpoint.
    #
    #    yaw_offset = wrap_to_pi(yaw - yaw_setpoint)
    #    yaw_offset = clamp(yaw_offset, YAW_LIMIT)
    #
    #  STEP 4: your Task 2A control law, with yaw_offset wherever Task 2A
    #  used yaw. Written exactly the way each Task 2A Tuner did it, so your
    #  gains work unchanged:
    #
    #  LQR:
    #    U = -(K[0] * tilt_dot + K[1] * tilt + K[2] * yaw_dot + K[3] * yaw_offset)
    #    sim.setJointTargetVelocity(front_motor, MOTOR_SIGN * U)
    #
    #  PID:
    #    tilt_integral += tilt * dt
    #    yaw_integral  += yaw_offset * dt
    #    U = (Kp_tilt * tilt + Ki_tilt * tilt_integral + Kd_tilt * tilt_dot) \
    #        - (Kp_yaw * yaw_offset + Ki_yaw * yaw_integral + Kd_yaw * yaw_dot)
    #    sim.setJointTargetVelocity(front_motor, -MOTOR_SIGN * U)
    #
    #  PID only: the yaw integral fills up during a turn. Clamp it, reset it
    #  when the setpoint changes, or keep Ki_yaw at 0.

    #################################


# =============================================================================
def sysCall_cleanup():
    # Runs ONCE, when the simulation ends. Stop the motor.
    sim.setJointTargetVelocity(front_motor, 0)

# =============================================================================
#  STUCK? CHECK THESE FIRST
# =============================================================================
#    Falls instantly and hard            -> MOTOR_SIGN
#    Balances at 0 but falls on a turn   -> yaw_offset not limited (YAW_LIMIT)
#    Spins the long way round            -> yaw_offset not wrapped
#    Stops short of each setpoint        -> yaw gain too weak, or YAW_LIMIT too small
#    Wobble that grows after each turn   -> yaw integral wind-up
#    Nothing turns                       -> YAW_LIMIT is still 0
#
#  Print before you tune:
#    print(f"sp={yaw_setpoint:+.3f} yaw={yaw:+.3f} tilt={tilt:+.3f} U={U:+.3f}")
# =============================================================================
