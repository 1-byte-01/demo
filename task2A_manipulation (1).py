#!/usr/bin/env python3


'''
*****************************************************************************************
*
*        		===============================================
*           		        StrataCobot (SC) Theme (eYRC 2026-27)
*        		===============================================
*
*  This script should be used to implement Task 2A (manipulation) of StrataCobot (SC) Theme (eYRC 2026-27).
*
*  This software is made available on an "AS IS WHERE IS BASIS".
*  Licensee/end user indemnifies and will keep e-Yantra indemnified from
*  any and all claim(s) that emanate from the use of the Software or
*  breach of the terms of this agreement.
*
*****************************************************************************************
'''

# Team ID:          [ 6382 ]
# Author List:		[ Jerson Miranda, Simran Katie Afonso, Flavia Melisa Oliveira, Ashley Snedon Fernandes ]
# Filename:		    task2A_manipulation.py
# Functions:
#			        [ quat_mul, quat_conj, tool_axis, __init__, tcpposecb, jointstatecb, armstatuscb,
#			        forcecb, switch_controller, call_magnet, find_ores, ore_position, build_plan,
#			        publish_zero_jog, publish_zero_twist, hold, servo, elapsed, jump, run_step,
#			        align_tool, recover, process_waypoints, main ]
# Nodes:		    arm_waypoints_node
#			        Publishing Topics  - [ /delta_twist_cmds, /delta_joint_cmds ]
#                   Subscribing Topics - [ /tcp_pose_raw, /joint_states, /arm_status, /ur7e/end_effector/force_magnitude, /tf, /tf_static ]
#                   Service Clients    - [ /magnet, /ur_arm_controller/switch_controller, /ur_arm_controller/unlock_protective_stop ]


################### IMPORT MODULES #######################

import rclpy
import sys
import math
import tf2_ros
from rclpy.node import Node
from control_msgs.msg import JointJog
from controller_manager_msgs.srv import SwitchController
from geometry_msgs.msg import PoseStamped, TwistStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Int32, Float32
from std_srvs.srv import SetBool, Trigger


##################### TASK CONSTANTS #######################

# ---- from Task 1B -------------------------------------------------------------------
unfold_target = [0.0, -1.20, -1.60, -1.90, 1.57, 1.57]
unfold_cap_rps = 0.25      # the servo's cap is 0.60 rad/s per joint; stay well under it
unfold_tolerance = 0.03    # rad, per joint

servo_ns = '/ur_arm_controller'
twist_controller = 'delta_twist_controller'
joint_controller = 'delta_joint_controller'

base_frame = 'base_link'

joint_names = [
    'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
    'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint',
]

cap_linear_mps = 0.15      # the servo's cap is 0.25 m/s on the MAGNITUDE; a message over it is dropped whole
cap_angular_rps = 0.35     # the servo's cap is 0.60 rad/s

# ---- ores (Magnetic Tool manual) -----------------------------------------------------
ore_types = ['azurite_ore', 'malachite_ore', 'vanadinite_ore']
ore_ids = (1, 2)
expected_ore_count = 4
scan_timeout_s = 20.0            # give up waiting for the full set of ore transforms after this
ore_half_height = 0.0381         # ore is 101.6 x 101.6 x 76.2 mm; perception publishes its MIDDLE
max_reach_xy = 0.85              # skip any ore further from the arm base than this (m)

# ---- containers (base_link, metres) --------------------------------------------------
# !! STILL TO CONFIRM !!  Inferred from the Task 1B waypoints 3-5 (0.333 m apart on y = +0.528) and the
# manuals: the cargo rover is to the arm's LEFT (+y) and azurite is the container nearest the arm.
# Check each with tcp_pose_raw while the tool is over the container.
container_xy = {
    'azurite_ore':    (-0.0763, 0.528),
    'malachite_ore':  (-0.4097, 0.528),
    'vanadinite_ore': (-0.7430, 0.528),
}
drop_offsets = [(0.0, 0.0), (0.0, -0.06)]    # 2nd ore of a kind lands beside the 1st, not on it
drop_z = 0.26                    # tool height over a container when the magnet is released (tune)
container_travel_z = 0.32        # tool height while moving over the containers
container_side_y = 0.15          # tool further left than this counts as being on the container side

# ---- routes ----------------------------------------------------------------------------
# The crusher is BEHIND the base station (the camera looks at it from -X). The Task 1B route went round
# the back (its waypoint 2) and would run the forearm into the crusher, so the arm now crosses IN FRONT
# of its base, at a radius that keeps it clear of the base singularity.
travel_z = 0.40                  # all long moves are made at least this high
entry_via = [(0.36, -0.35, 0.45), (0.05, -0.55, 0.40)]    # unfolded pose -> conveyor side (from Task 1B)
transit_via = [(0.45, -0.30, 0.40), (0.45, 0.30, 0.40)]   # conveyor side -> container side (reversed to come back)
via_tolerance = 0.05
min_radius_xy = 0.25             # warn about any target closer to the base axis than this

# ---- motion ---------------------------------------------------------------------------
control_rate = 0.02              # s
kp = 0.8
speed_travel = 0.10
speed_approach = 0.05
speed_lift = 0.05
speed_contact = 0.015            # final centimetres onto the ore; the arm coasts on after the last command
min_speed = 0.01
max_dv = 0.002                   # m/s per tick ramp
angular_gain = 1.5
angular_cap = 0.25
move_tolerance = 0.015
align_tolerance = 0.007          # the 81 mm magnet face must sit inside the 101.6 mm ore face: +/-10 mm at most

# ---- tool orientation -------------------------------------------------------------------
# The magnet must come straight down on the ore's top face (faces parallel). The tool's own axis is
# taken to be the tcp frame's +Z; if the first log line shows the tool axis NOT near (0, 0, -1) while the
# tool is visibly pointing down, change tool_axis_local.
tool_axis_local = (0.0, 0.0, 1.0)
align_auto_max_rad = 0.35        # only auto-correct a tilt up to this; more means the axis guess is wrong
align_done_rad = 0.03
align_timeout_s = 10.0

# ---- picking --------------------------------------------------------------------------
hover_clearance = 0.12           # tool height above the ore's top face while lining up
tool_contact_offset = 0.0        # tcp height above the ore's top face at touch (the tcp is on the magnet face)
magnet_on_dist = 0.04            # energise the coil this far above the top face
spring_press = 0.008             # descent target is this far BELOW the top face (tool spring has 10 mm travel)
settle_s = 0.8
release_s = 1.0
contact_timeout_s = 12.0
max_pick_retries = 2
max_ore_attempts = 2             # after a protective stop an ore is retried once

# Force at the tool in newtons, tool weight already removed (Magnetic Tool manual):
#   free air ~ 0     holding an ore ~ 1.47     holding the package ~ 2.94
contact_force_stop = 0.30        # touched: stop descending (the spring force rises with travel)
held_force_min = 0.75            # above this after the lift the ore is on the magnet
released_force_max = 0.40        # below this after the release the ore has let go

# ---- safety --------------------------------------------------------------------------
protective_min, protective_max = 20, 23       # latching protective stops
retreat_speed = 0.04
recovery_window_s = 4.5          # the servo gives 5 s of COMMANDING after an unlock


##################### HELPER FUNCTIONS #######################

def quat_mul(a, b):
    '''
    Description:    Hamilton product of two quaternions given as (x, y, z, w).
    '''
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz)


def quat_conj(q):
    '''
    Description:    Conjugate (inverse, for a unit quaternion) of (x, y, z, w).
    '''
    return (-q[0], -q[1], -q[2], q[3])


def tool_axis(q):
    '''
    Description:    Direction, in base_link, of the tool's own axis for the orientation q (x, y, z, w).
    '''
    x, y, z, w = q
    ax, ay, az = tool_axis_local
    # rotation matrix applied to the local axis
    return (
        (1 - 2 * (y * y + z * z)) * ax + 2 * (x * y - w * z) * ay + 2 * (x * z + w * y) * az,
        2 * (x * y + w * z) * ax + (1 - 2 * (x * x + z * z)) * ay + 2 * (y * z - w * x) * az,
        2 * (x * z - w * y) * ax + 2 * (y * z + w * x) * ay + (1 - 2 * (x * x + y * y)) * az,
    )


##################### CLASS DEFINITION #######################

class arm_waypoints(Node):
    '''
    ___CLASS___

    Description:    Class which serves the purpose to pick each ore off the conveyor with the
                    magnet and drop it in the container for its kind, driving the tool with the
                    arm's velocity command interfaces.
    '''

    def __init__(self):
        '''
        Description:    Initialization of class arm_waypoints
        '''

        super().__init__(
            'arm_waypoints_node',
            parameter_overrides=[rclpy.parameter.Parameter(
                'use_sim_time', rclpy.Parameter.Type.BOOL, True)])

        ############ Topic PUBLISHERS ############

        self.twist_pub = self.create_publisher(TwistStamped, '/delta_twist_cmds', 10)
        self.joint_pub = self.create_publisher(JointJog, '/delta_joint_cmds', 10)

        ############ Topic SUBSCRIPTIONS ############

        self.tcp_sub = self.create_subscription(PoseStamped, '/tcp_pose_raw', self.tcpposecb, 20)
        self.joint_sub = self.create_subscription(JointState, '/joint_states', self.jointstatecb, 50)
        self.status_sub = self.create_subscription(Int32, '/arm_status', self.armstatuscb, 10)
        self.force_sub = self.create_subscription(
            Float32, '/ur7e/end_effector/force_magnitude', self.forcecb, 10)

        ############ Constructor VARIABLES/OBJECTS ############

        self.switch_cli = self.create_client(SwitchController, f'{servo_ns}/switch_controller')
        self.unlock_cli = self.create_client(Trigger, f'{servo_ns}/unlock_protective_stop')
        self.magnet_cli = self.create_client(SetBool, '/magnet')
        self.tf_buffer = tf2_ros.buffer.Buffer()
        self.listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.tcp_pose = None
        self.joint_angles = None
        self.arm_status = None
        self.force = None

        self.desired_orientation = None
        self.prev_v = (0.0, 0.0, 0.0)

        self.stage = 'unfold'            # unfold -> switching -> align -> scan -> run -> done   (+ recover)
        self.switch_future = None
        self.scan_start = None
        self.align_start = None
        self.remaining = []              # ore frame names still to pick
        self.attempts = {}               # ore frame -> times started
        self.plan = []                   # list of step tuples for the current ore
        self.pc = 0                      # index of the current step
        self.sstate = {}                 # scratch space of the current step
        self.cur_name = None
        self.cur_kind = None
        self.first_ore = True
        self.retries = 0
        self.dropped = {}                # kind -> ores already dropped in its container
        self.picked = 0
        self.magnet_is_on = False
        self.rec = None                  # recovery bookkeeping
        self.resume_stage = None
        self.stops = 0

        self.get_logger().info('Task 2A manipulation node initialized.')
        for kind, (cx, cy) in container_xy.items():
            self.get_logger().info(f'Container for {kind}: ({cx:.3f}, {cy:.3f})')

        # make the joint interface live for the unfold; twist servoing is the default mode
        ready = False
        for attempt in range(3):
            try:
                if self.switch_controller(joint_controller):
                    ready = True
                    break
            except Exception as e:
                self.get_logger().warn(f'Switch attempt {attempt + 1} failed: {e}')
        if not ready:
            self.get_logger().warn('Switch failed; continuing, twist servoing is the default mode.')

        self.timer = self.create_timer(control_rate, self.process_waypoints)


    ##################### CALLBACKS #######################

    def tcpposecb(self, data):
        '''
        Description:    Callback function for the tool pose topic.

        Args:
            data (PoseStamped):    Pose of the tool, reported in base_link

        Returns:
        '''

        self.tcp_pose = data.pose


    def jointstatecb(self, data):
        '''
        Description:    Callback function for the joint states topic. Angles are matched BY NAME,
                        the topic is in alphabetical order.

        Args:
            data (JointState):    Joint feedback published by the arm

        Returns:
        '''

        try:
            self.joint_angles = [data.position[data.name.index(j)] for j in joint_names]
        except ValueError:
            self.joint_angles = None


    def armstatuscb(self, data):
        '''
        Description:    Callback function for the arm status topic.

        Args:
            data (Int32):    One state code describing what the arm is doing

        Returns:
        '''

        if data.data != self.arm_status and data.data != 0:
            self.get_logger().warn(f'Arm status changed: {data.data}')
        self.arm_status = data.data


    def forcecb(self, data):
        '''
        Description:    Callback function for the force at the tool, in newtons, tool weight removed.

        Args:
            data (Float32):    Force magnitude

        Returns:
        '''

        self.force = float(data.data)


    ##################### SERVICES #######################

    def switch_controller(self, controller):
        '''
        Description:    Make one of the arm's two command interfaces the active one.
                        Only call from __init__ (it spins); the timer polls futures instead.

        Args:
            controller  (str):      'delta_twist_controller' or 'delta_joint_controller'

        Returns:
            success     (bool):     Whether the controller was activated
        '''

        if controller == twist_controller:
            activate, deactivate = twist_controller, joint_controller
        elif controller == joint_controller:
            activate, deactivate = joint_controller, twist_controller
        else:
            self.get_logger().error(f'Unknown controller requested: {controller}')
            return False

        if not self.switch_cli.wait_for_service(timeout_sec=20.0):
            self.get_logger().error('Switch controller service is not available.')
            return False

        req = SwitchController.Request()
        req.activate_controllers = [activate]
        req.deactivate_controllers = [deactivate]
        req.strictness = SwitchController.Request.STRICT

        future = self.switch_cli.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)

        if not future.done():
            self.get_logger().error('Controller switch service timed out.')
            return False
        try:
            response = future.result()
        except Exception as e:
            self.get_logger().error(f'Controller switch service call failed: {e}')
            return False
        if response is not None and response.ok:
            self.get_logger().info(f'Controller switched to: {activate}')
            return True
        self.get_logger().error(f'Controller switch rejected: {activate}')
        return False


    def call_magnet(self, on):
        '''
        Description:    Ask the magnet service to switch on or off, without blocking.
                        The reply only confirms the call arrived; the force says what is held.

        Args:
            on          (bool):     True to energise, False to release

        Returns:
            future      (Future):   Poll future.done(); None if the service is not up yet
        '''

        if not self.magnet_cli.service_is_ready():
            return None
        req = SetBool.Request()
        req.data = bool(on)
        self.magnet_is_on = bool(on)
        return self.magnet_cli.call_async(req)


    ##################### PLANNING #######################

    def find_ores(self):
        '''
        Description:    Which ore transforms exist right now.

        Returns:
            names       (list):     Frame names such as 'azurite_ore_1' that resolve against base_link
        '''

        names = []
        for kind in ore_types:
            for oid in ore_ids:
                name = f'{kind}_{oid}'
                if self.tf_buffer.can_transform(base_frame, name, rclpy.time.Time()):
                    names.append(name)
        return names


    def ore_position(self, name):
        '''
        Description:    Latest position of an ore frame in base_link, or None.
        '''

        try:
            t = self.tf_buffer.lookup_transform(base_frame, name, rclpy.time.Time()).transform.translation
        except Exception:
            return None
        return (t.x, t.y, t.z)


    def build_plan(self, name, pos):
        '''
        Description:    Build the list of steps that picks one ore and drops it in its container.
                        Every long move is made at travel height and every approach to the ore
                        or container is a vertical descent, so the arm never sweeps low.
                        Steps: ('log', text) ('label', n) ('move', xyz, tol, speed) ('rise', z)
                        ('contact', xyz, speed) ('magnet', bool) ('wait', s) ('baseline',)
                        ('check_held', retry_label, skip_label) ('check_released',)

        Args:
            name        (str):      Ore frame, e.g. 'malachite_ore_1'
            pos         (tuple):    The ore's middle in base_link

        Returns:
            plan        (list)
        '''

        kind = name.rsplit('_', 1)[0]
        ox, oy, oz = pos
        top = oz + ore_half_height                          # the face the magnet lands on
        above = (ox, oy, max(travel_z, top + hover_clearance))
        hover = (ox, oy, top + hover_clearance)
        near = (ox, oy, top + tool_contact_offset + magnet_on_dist)
        press = (ox, oy, top + tool_contact_offset - spring_press)
        dx, dy = drop_offsets[min(self.dropped.get(kind, 0), len(drop_offsets) - 1)]
        cx, cy = container_xy[kind][0] + dx, container_xy[kind][1] + dy
        over_box = (cx, cy, container_travel_z)
        drop = (cx, cy, drop_z)

        if math.hypot(ox, oy) < min_radius_xy:
            self.get_logger().warn(f'{name} is only {math.hypot(ox, oy):.2f} m from the base axis.')

        p = self.tcp_pose.position
        plan = [('log', f'Picking {name}: middle at ({ox:.3f}, {oy:.3f}, {oz:.3f}), top face z={top:.3f}'),
                ('rise', travel_z)]

        # get to the conveyor side, high, and from the right place
        if self.first_ore:
            plan += [('move', v, via_tolerance, speed_travel) for v in entry_via]
        elif p.y > container_side_y:
            plan += [('move', v, via_tolerance, speed_travel) for v in reversed(transit_via)]

        plan += [
            ('move', above, move_tolerance, speed_travel),
            ('label', 'retry'),
            ('magnet', False),
            ('wait', 0.2),
            ('move', hover, align_tolerance, speed_approach),        # straight down, lined up
            ('baseline',),
            ('move', near, align_tolerance, speed_approach),
            ('magnet', True),
            ('contact', press, speed_contact),
            ('wait', settle_s),
            ('move', hover, move_tolerance, speed_lift),
            ('wait', 0.5),
            ('check_held', 'retry', 'skip'),
            ('move', above, move_tolerance, speed_lift),
        ]
        plan += [('move', v, via_tolerance, speed_travel) for v in transit_via]
        plan += [
            ('move', over_box, move_tolerance, speed_travel),
            ('move', drop, align_tolerance, speed_approach),
            ('magnet', False),
            ('wait', release_s),
            ('check_released',),
            ('move', over_box, move_tolerance, speed_lift),
            ('label', 'skip'),
            ('magnet', False),
            ('wait', 0.3),
        ]
        return plan


    ##################### MOTION PRIMITIVES #######################

    def publish_zero_jog(self):
        '''
        Description:    Publishes a zero joint velocity command.
        '''

        jog = JointJog()
        jog.header.stamp = self.get_clock().now().to_msg()
        jog.joint_names = joint_names
        jog.velocities = [0.0] * 6
        self.joint_pub.publish(jog)


    def publish_zero_twist(self):
        '''
        Description:    Publishes a zero velocity twist. Used for every hold, because the arm stops
                        0.15 s after the last message it received.
        '''

        self.prev_v = (0.0, 0.0, 0.0)
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = base_frame
        self.twist_pub.publish(msg)


    def hold(self):
        '''
        Description:    Stand still this tick (zero twist is still published).
        '''

        self.publish_zero_twist()


    def orientation_rate(self):
        '''
        Description:    Angular velocity that returns the tool to the saved orientation.

        Returns:
            (wx, wy, wz)  (tuple):  rad/s about base_link axes, capped
        '''

        q = self.tcp_pose.orientation
        e = quat_mul(self.desired_orientation, quat_conj((q.x, q.y, q.z, q.w)))
        if e[3] < 0.0:
            e = (-e[0], -e[1], -e[2], -e[3])
        wx, wy, wz = (angular_gain * 2.0 * e[0], angular_gain * 2.0 * e[1], angular_gain * 2.0 * e[2])
        w = math.sqrt(wx * wx + wy * wy + wz * wz)
        if w > angular_cap:
            wx, wy, wz = wx * angular_cap / w, wy * angular_cap / w, wz * angular_cap / w
        return wx, wy, wz


    def servo(self, target, tol, speed):
        '''
        Description:    One control tick toward a position, holding the saved orientation.
                        Velocity = Kp * error, the WHOLE vector scaled to the speed limit.

        Args:
            target      (tuple):    Goal (x, y, z) in base_link
            tol         (float):    Arrival distance (m)
            speed       (float):    Speed limit for this move (m/s)

        Returns:
            reached     (bool):     True when within tol (a zero twist was published)
        '''

        p = self.tcp_pose.position
        ex, ey, ez = target[0] - p.x, target[1] - p.y, target[2] - p.z
        dist = math.sqrt(ex * ex + ey * ey + ez * ez)

        if dist <= tol:
            self.hold()
            return True

        vx, vy, vz = kp * ex, kp * ey, kp * ez
        mag = math.sqrt(vx * vx + vy * vy + vz * vz)
        limit = min(speed, cap_linear_mps)
        if mag > limit:
            s = limit / mag
        elif mag < min_speed:
            s = min_speed / mag
        else:
            s = 1.0
        vx, vy, vz = vx * s, vy * s, vz * s

        px, py, pz = self.prev_v
        vx = px + max(-max_dv, min(max_dv, vx - px))
        vy = py + max(-max_dv, min(max_dv, vy - py))
        vz = pz + max(-max_dv, min(max_dv, vz - pz))
        self.prev_v = (vx, vy, vz)

        wx, wy, wz = self.orientation_rate()

        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = base_frame
        msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z = vx, vy, vz
        msg.twist.angular.x, msg.twist.angular.y, msg.twist.angular.z = wx, wy, wz
        self.twist_pub.publish(msg)
        return False


    ##################### STEP INTERPRETER #######################

    def elapsed(self):
        '''
        Description:    Seconds since the current step began (sim clock).
        '''

        return (self.get_clock().now() - self.sstate['t0']).nanoseconds / 1e9


    def jump(self, label):
        '''
        Description:    Move the step pointer to a label in the current plan.
        '''

        for i, s in enumerate(self.plan):
            if s[0] == 'label' and s[1] == label:
                self.pc = i
                self.sstate = {}
                return
        self.pc = len(self.plan)
        self.sstate = {}


    def run_step(self):
        '''
        Description:    Execute one tick of the current step.

        Returns:
            finished    (bool):     True when the step is complete and the pointer should advance
        '''

        step = self.plan[self.pc]
        kind = step[0]
        if 't0' not in self.sstate:
            self.sstate['t0'] = self.get_clock().now()

        if kind == 'log':
            self.get_logger().info(step[1])
            return True

        if kind == 'label':
            return True

        if kind == 'move':
            return self.servo(step[1], step[2], step[3])

        if kind == 'rise':
            # straight up to at least this height from wherever the tool is
            p = self.tcp_pose.position
            if p.z >= step[1] - 0.02:
                self.hold()
                return True
            return self.servo((p.x, p.y, step[1]), 0.02, speed_lift)

        if kind == 'wait':
            self.hold()
            return self.elapsed() >= step[1]

        if kind == 'baseline':
            # free-air force with the magnet off; it should read about 0 N
            self.hold()
            self.sstate.setdefault('vals', [])
            if self.force is not None:
                self.sstate['vals'].append(self.force)
            if self.elapsed() >= 0.5:
                v = self.sstate['vals']
                f = sum(v) / len(v) if v else 0.0
                self.get_logger().info(f'Free-air force {f:.3f} N (expect ~0).')
                return True
            return False

        if kind == 'magnet':
            # non-blocking: issue the call, then poll the future
            self.hold()
            fut = self.sstate.get('fut')
            if fut is None:
                fut = self.call_magnet(step[1])
                if fut is None:
                    if self.elapsed() > 10.0:
                        self.get_logger().error('Magnet service not available.')
                        return True
                    return False
                self.sstate['fut'] = fut
                self.sstate['call_t'] = self.get_clock().now()
                return False
            if fut.done():
                try:
                    fut.result()
                    self.get_logger().info(f'Magnet {"ON" if step[1] else "OFF"}')
                except Exception as e:
                    self.get_logger().error(f'Magnet call failed: {e}')
                return True
            if (self.get_clock().now() - self.sstate['call_t']).nanoseconds / 1e9 > 3.0:
                self.get_logger().warn('Magnet call timed out; continuing.')
                return True
            return False

        if kind == 'contact':
            # descend slowly; stop on touch (force or a collision warning), reaching the
            # target, or timeout. The target is inside the tool spring's travel.
            f = self.force if self.force is not None else 0.0
            if f > contact_force_stop or self.arm_status == 12:
                self.get_logger().info(f'Touch: force {f:.2f} N')
                self.hold()
                return True
            if self.elapsed() > contact_timeout_s:
                self.get_logger().warn('Contact timeout.')
                self.hold()
                return True
            return self.servo(step[1], 0.003, step[2])

        if kind == 'check_held':
            self.hold()
            f = self.force
            if f is None:
                self.get_logger().warn('No force reading; assuming the ore is held.')
                self.retries = 0
                return True
            if f > held_force_min:
                self.get_logger().info(f'{self.cur_name} held (force {f:.2f} N).')
                self.retries = 0
                return True
            if self.retries < max_pick_retries:
                self.retries += 1
                self.get_logger().warn(f'{self.cur_name} not held (force {f:.2f} N); retry {self.retries}.')
                self.jump(step[1])
            else:
                self.get_logger().error(f'{self.cur_name} not picked; skipping it.')
                self.retries = 0
                self.jump(step[2])
            return False

        if kind == 'check_released':
            self.hold()
            if self.elapsed() < 0.2:
                return False
            f = self.force if self.force is not None else 0.0
            if f > released_force_max:
                self.get_logger().warn(f'Force still {f:.2f} N after release.')
            else:
                self.get_logger().info(f'{self.cur_name} released.')
            self.picked += 1
            self.dropped[self.cur_kind] = self.dropped.get(self.cur_kind, 0) + 1
            return True

        self.get_logger().error(f'Unknown step {step}')
        return True


    ##################### STAGES #######################

    def align_tool(self):
        '''
        Description:    Turn the tool until its axis points straight down, so the magnet face meets
                        the ore's top face flat. Only corrects a small tilt; a large one means the
                        axis guess (tool_axis_local) is wrong, and then the orientation is just held.

        Returns:
            done        (bool):     True when the stage is over
        '''

        q = self.tcp_pose.orientation
        a = tool_axis((q.x, q.y, q.z, q.w))
        b = (0.0, 0.0, -1.0)
        cx_ = a[1] * b[2] - a[2] * b[1]
        cy_ = a[2] * b[0] - a[0] * b[2]
        cz_ = a[0] * b[1] - a[1] * b[0]
        s = math.sqrt(cx_ * cx_ + cy_ * cy_ + cz_ * cz_)
        ang = math.atan2(s, a[0] * b[0] + a[1] * b[1] + a[2] * b[2])

        if 'logged' not in self.sstate:
            self.sstate['logged'] = True
            self.get_logger().info(
                f'Tool axis = ({a[0]:.2f}, {a[1]:.2f}, {a[2]:.2f}); {math.degrees(ang):.1f} deg from straight down.')

        t = (self.get_clock().now() - self.align_start).nanoseconds / 1e9
        if ang < align_done_rad or t > align_timeout_s or ang > align_auto_max_rad:
            if ang > align_auto_max_rad:
                self.get_logger().warn(
                    'Tool axis is far from vertical: not auto-correcting. If the tool is really pointing '
                    'down, change tool_axis_local; if not, the unfold pose needs another wrist angle.')
            self.publish_zero_twist()
            self.desired_orientation = (q.x, q.y, q.z, q.w)
            return True

        w = min(angular_cap, 1.5 * ang)
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = base_frame
        msg.twist.angular.x = w * cx_ / s
        msg.twist.angular.y = w * cy_ / s
        msg.twist.angular.z = w * cz_ / s
        self.twist_pub.publish(msg)
        return False


    def recover(self):
        '''
        Description:    After a latching protective stop: release the magnet, unlock the arm, then
                        stream an upward twist (the servo only accepts motion that reduces the
                        violation) until the status clears. The current ore is requeued once.
        '''

        r = self.rec
        now = self.get_clock().now()

        if r['phase'] == 'unlock':
            self.publish_zero_twist()
            if r.get('fut') is None:
                if self.magnet_is_on:
                    self.call_magnet(False)
                if not self.unlock_cli.service_is_ready():
                    return
                r['fut'] = self.unlock_cli.call_async(Trigger.Request())
                r['t0'] = now
                return
            if r['fut'].done():
                try:
                    self.get_logger().warn(f'Unlock: {r["fut"].result().message}')
                except Exception as e:
                    self.get_logger().error(f'Unlock failed: {e}')
                r['phase'] = 'retreat'
                r['t0'] = now
                r['cmd_ticks'] = 0
            return

        if r['phase'] == 'retreat':
            if self.arm_status is not None and self.arm_status < protective_min:
                r['phase'] = 'settle'
                r['t0'] = now
                self.publish_zero_twist()
                return
            if r['cmd_ticks'] * control_rate > recovery_window_s:
                self.get_logger().error('Could not clear the protective stop. Giving up.')
                self.stage = 'done'
                return
            r['cmd_ticks'] += 1
            msg = TwistStamped()
            msg.header.stamp = now.to_msg()
            msg.header.frame_id = base_frame
            msg.twist.linear.z = retreat_speed
            self.twist_pub.publish(msg)
            return

        # settle
        self.publish_zero_twist()
        if (now - r['t0']).nanoseconds / 1e9 > 0.5:
            self.get_logger().warn('Protective stop cleared; carrying on.')
            if self.resume_stage == 'run' and self.cur_name and self.plan and self.pc < len(self.plan):
                if self.attempts.get(self.cur_name, 0) < max_ore_attempts:
                    self.remaining.insert(0, self.cur_name)
                self.plan, self.pc, self.sstate = [], 0, {}
            self.rec = None
            self.prev_v = (0.0, 0.0, 0.0)
            self.stage = self.resume_stage


    ##################### MAIN TIMER #######################

    def process_waypoints(self):
        '''
        Description:    Timer function: unfold the arm, aim the tool down, find the ores, then pick
                        and drop them one at a time. Publishes exactly one command on every tick.

        Args:
        Returns:
        '''

        if (self.tcp_pose is None or self.joint_angles is None
                or self.arm_status is None):
            return

        st = self.arm_status

        # tier 9: the arm itself is not ready or is broken; stand still
        if st >= 90:
            self.publish_zero_twist()
            self.get_logger().warn(f'Arm fault {st}: waiting.', throttle_duration_sec=2.0)
            return

        # tier 2: a latching protective stop
        if protective_min <= st <= protective_max and self.stage not in ('recover', 'done'):
            self.stops += 1
            self.get_logger().error(f'PROTECTIVE STOP {st} (stop #{self.stops}); recovering.')
            self.publish_zero_twist()
            self.resume_stage = self.stage
            self.stage = 'recover'
            self.rec = {'phase': 'unlock'}
            return

        if self.stage == 'recover':
            self.recover()
            return

        # stage 1: unfold with joint velocities
        if self.stage == 'unfold':
            err = [t - a for t, a in zip(unfold_target, self.joint_angles)]
            if max(abs(e) for e in err) < unfold_tolerance:
                self.publish_zero_jog()
                req = SwitchController.Request()
                req.activate_controllers = [twist_controller]
                req.deactivate_controllers = [joint_controller]
                req.strictness = SwitchController.Request.STRICT
                self.switch_future = self.switch_cli.call_async(req)
                self.stage = 'switching'
                self.get_logger().info('Unfold done, switching to twist servoing.')
                return
            vel = [1.5 * e for e in err]
            biggest = max(abs(v) for v in vel)
            if biggest > unfold_cap_rps:
                vel = [v * unfold_cap_rps / biggest for v in vel]
            jog = JointJog()
            jog.header.stamp = self.get_clock().now().to_msg()
            jog.joint_names = joint_names
            jog.velocities = vel
            self.joint_pub.publish(jog)
            self.get_logger().info(
                f'unfolding joints={[round(a, 2) for a in self.joint_angles]}',
                throttle_duration_sec=0.5)
            return

        if self.stage == 'switching':
            self.publish_zero_jog()
            if self.switch_future.done():
                res = self.switch_future.result()
                if res is not None and res.ok:
                    q = self.tcp_pose.orientation
                    self.desired_orientation = (q.x, q.y, q.z, q.w)
                    self.stage = 'align'
                    self.align_start = self.get_clock().now()
                    self.sstate = {}
                    self.get_logger().info('Twist servoing active; aligning the tool.')
                else:
                    self.stage = 'unfold'   # try again
            return

        # stage 2b: tool pointing straight down
        if self.stage == 'align':
            if self.align_tool():
                self.stage = 'scan'
                self.scan_start = self.get_clock().now()
                self.sstate = {}
            return

        # stage 3: wait for the perception node's transforms to settle
        if self.stage == 'scan':
            self.publish_zero_twist()
            found = self.find_ores()
            waited = (self.get_clock().now() - self.scan_start).nanoseconds / 1e9
            if (len(found) >= expected_ore_count and waited > 3.0) or (found and waited > scan_timeout_s):
                self.remaining = found
                self.stage = 'run'
                self.get_logger().info(f'Ores found: {found}')
            else:
                self.get_logger().info(f'Waiting for ore transforms ({len(found)} so far)...',
                                       throttle_duration_sec=2.0)
            return

        # stage 4: pick and drop
        if self.stage == 'run':
            if self.pc >= len(self.plan):
                # the next ore: nearest to the tool among those left
                p = self.tcp_pose.position
                best = None
                for name in list(self.remaining):
                    pos = self.ore_position(name)
                    if pos is None or math.hypot(pos[0], pos[1]) > max_reach_xy:
                        self.get_logger().warn(f'Skipping {name}: no position or out of reach.')
                        self.remaining.remove(name)
                        continue
                    d = math.hypot(pos[0] - p.x, pos[1] - p.y)
                    if best is None or d < best[0]:
                        best = (d, name, pos)
                if best is None:
                    self.stage = 'done'
                    self.get_logger().info(
                        f'All ores handled ({self.picked} dropped, {self.stops} protective stops).')
                    self.publish_zero_twist()
                    return
                _, self.cur_name, pos = best
                self.remaining.remove(self.cur_name)
                self.attempts[self.cur_name] = self.attempts.get(self.cur_name, 0) + 1
                self.cur_kind = self.cur_name.rsplit('_', 1)[0]
                self.plan = self.build_plan(self.cur_name, pos)
                self.first_ore = False
                self.pc = 0
                self.sstate = {}
                self.retries = 0

            # bookkeeping steps take no time: run them now so this tick still sends a command
            while self.pc < len(self.plan) and self.plan[self.pc][0] in ('log', 'label'):
                self.run_step()
                self.pc += 1
                self.sstate = {}
            if self.pc >= len(self.plan):
                self.publish_zero_twist()
                return

            p = self.tcp_pose.position
            self.get_logger().info(
                f'{self.cur_name} step {self.pc}/{len(self.plan)} {self.plan[self.pc][0]} '
                f'target={self.plan[self.pc][1] if len(self.plan[self.pc]) > 1 else "-"} '
                f'tcp=({p.x:.3f},{p.y:.3f},{p.z:.3f}) force={self.force} status={self.arm_status}',
                throttle_duration_sec=0.5)

            if self.run_step():
                self.pc += 1
                self.sstate = {}
            return

        # done: keep the dead-man switch fed
        self.publish_zero_twist()


##################### FUNCTION DEFINITION #######################

def main():
    '''
    Description:    Main function which creates a ROS node and spins around for the
                    arm_waypoints class to perform its task
    '''

    rclpy.init(args=sys.argv)

    arm_waypoints_class = arm_waypoints()

    rclpy.spin(arm_waypoints_class)

    arm_waypoints_class.destroy_node()

    rclpy.shutdown()


if __name__ == '__main__':

    main()
