from math import sin, cos, pi, sqrt, atan, asin, acos, atan2
import numpy as np
from abc import abstractmethod

def wrap_to_minus_pi_to_zero(angle: float) -> float:
        """Wraps the angle to the range [-pi, 0].
        
        The logic is:
        If angle < -pi, add pi;
        if angle > 0, subtract pi;
        if angle > 0, subtract pi again. (Means originally angle was > pi)
        """
        if angle < -pi:
             return angle + pi
        elif angle > 0:
            if angle > pi:
                return angle - 2*pi
            else:
                return angle - pi
        return angle  # Already in the range [-pi, 0]

class GeneralBennettIK:
    def __init__(
            self,
            l, x0, y0, z0, zt, beta,
            thigh_offset=0.0, knee_offset=0.0,
            success_thresh = 1e-6,
        ):
        self.l = l  # Linkage length, same length for all links
        self.x0 = x0  # x pos of foot in initial configuration
        self.y0 = y0  # y pos of foot in initial configuration
        self.z0 = z0  # z pos of foot in initial configuration
        self.zt = zt  # z position of the knee joint origin
        self.beta = beta  # Bennett linkage beta in radians

        self.thigh_offset = thigh_offset  # Motor thigh angular offset in radians
        self.knee_offset = knee_offset  # Motor knee angular offset in radians

        self.success_thresh = success_thresh  # Threshold for success in IK

        self.sin_beta = sin(self.beta)
        self.cos_beta = cos(self.beta)

        self.A = -self.y0*cos(self.beta) - self.z0*sin(self.beta) + self.zt*sin(self.beta)
        self.B = -self.y0*sin(self.beta) + self.z0*cos(self.beta) - self.zt*cos(self.beta)
        self.C = (self.x0-self.l)
        self.D = self.y0*sin(self.beta)*cos(self.beta) - self.z0*cos(self.beta)**2 - self.zt*sin(self.beta)**2

        self.E = -self.zt*sin(self.beta)
        self.K = self.A**2 + self.B**2*sin(self.beta)**2 + self.C**2 + self.D**2 + self.l**2
        self.M = 2*(self.C*self.l + self.A*self.E)
        self.N = 2*(self.A*self.l - self.C*self.E)
        self.H = sqrt(self.M**2 + self.N**2)

        self.q3_single_sol = atan(-self.M/self.N)
        if self.q3_single_sol > 0:
            self.q3_single_sol -= pi

    def analytical_inverse_kinematics_span_tree(self, pos: list, reference_joint_pos: list | None = None) -> tuple[list, list]:
        ''' Analytical inverse kinematics for the Bennett linkage.
        Args:
            pos: Desired foot position in the hip frame, shape (N, 3).
            reference_joint_pos: Reference joint positions. When two valid solutions are found, \
                the one closest to the reference will be selected. If set to None, the reference will not be used. \
                the first valid solution will be returned.
        Returns:
            tuple: (best_solution, success, solutions, errors)
        '''
        
        # There are totally 8 possible solutions
        ik_solutions = [[0 for i in range(3)] for j in range(8)]

        x = pos[0]
        y = pos[1]
        z = pos[2]

        R = x**2 + y**2 + z**2
        S = R - self.K

        # Check if any solution is infeasible
        mns_square_sum = self.M**2 + self.N**2 - S**2
        if mns_square_sum < 0:
            return None, False, None, None

        # If no error raised, there are one or two solutions for q3
        q3_1 = 2 * atan( ( self.N + sqrt(mns_square_sum) ) / ( self.M + S ) )
        q3_2 = 2 * atan( ( self.N - sqrt(mns_square_sum) ) / ( self.M + S ) )
        q3_1 = wrap_to_minus_pi_to_zero(q3_1) # Wrap to [-pi, 0]
        q3_2 = wrap_to_minus_pi_to_zero(q3_2) # Wrap to [-pi, 0]

        # Depending on if M + S = 0, we have one or two solutions for q3
        if self.M == -S:
            # If M + S = 0, there is only one solution for q3
            q3_1 = self.q3_single_sol
            q3_2 = self.q3_single_sol

        for i in range(4):
            ik_solutions[i][2] = q3_1
        for i in range(4, 8):
            ik_solutions[i][2] = q3_2

        # When q3_1 is used
        # Solve for q2 and q1
        P = self.C*cos(q3_1) + self.A*sin(q3_1) + self.l
        Q = self.A*self.cos_beta*cos(q3_1) - \
            self.C*self.cos_beta*sin(q3_1) + \
            self.B*self.sin_beta
        U = self.A*self.sin_beta*cos(q3_1) - \
            self.C*self.sin_beta*sin(q3_1) + \
            self.D
        square_sum = y**2 + z**2 - U**2
        if square_sum < 0:
            # No valid solution when q3_1 is used
            # Leave the solutions as zero
            pass
        else:
            T = sqrt(square_sum)
            q2_1 = atan2(Q*x-P*T, P*x+Q*T)
            q2_1 = self.wrap_q2(q2_1)  # Wrap to [0, 2*pi]
            q2_2 = atan2(Q*x+P*T, P*x-Q*T)
            q2_2 = self.wrap_q2(q2_2)  # Wrap to [0, 2*pi]
            q1_1 = atan2(U*y+T*z, T*y-U*z)
            if q1_1 > 0:
                q1_1 -= pi
            q1_2 = atan2(U*y-T*z, -T*y-U*z)
            if q1_2 > 0:
                q1_2 -= pi

            ik_solutions[0][0] = q1_1
            ik_solutions[0][1] = q2_1
            ik_solutions[1][0] = q1_1
            ik_solutions[1][1] = q2_2
            ik_solutions[2][0] = q1_2
            ik_solutions[2][1] = q2_1
            ik_solutions[3][0] = q1_2
            ik_solutions[3][1] = q2_2

        # When q3_2 is used
        # Solve for q2 and q1
        P = self.C*cos(q3_2) + self.A*sin(q3_2) + self.l
        Q = self.A*self.cos_beta*cos(q3_2) - \
            self.C*self.cos_beta*sin(q3_2) + \
            self.B*self.sin_beta
        U = self.A*self.sin_beta*cos(q3_2) - \
            self.C*self.sin_beta*sin(q3_2) + \
            self.D
        square_sum = y**2 + z**2 - U**2
        if square_sum < 0:
            # No valid solution when q3_1 is used
            # Leave the solutions as zero
            pass
        else:
            T = sqrt(square_sum)
            q2_1 = atan2(Q*x-P*T, P*x+Q*T)
            q2_1 = self.wrap_q2(q2_1)  # Wrap to [0, 2*pi]
            q2_2 = atan2(Q*x+P*T, P*x-Q*T)
            q2_2 = self.wrap_q2(q2_2)  # Wrap to [0, 2*pi]
            q1_1 = atan2(U*y+T*z, T*y-U*z)
            if q1_1 > 0:
                q1_1 -= pi
            q1_2 = atan2(U*y-T*z, -T*y-U*z)
            if q1_2 > 0:
                q1_2 -= pi

            ik_solutions[4][0] = q1_1
            ik_solutions[4][1] = q2_1
            ik_solutions[5][0] = q1_1
            ik_solutions[5][1] = q2_2
            ik_solutions[6][0] = q1_2
            ik_solutions[6][1] = q2_1
            ik_solutions[7][0] = q1_2
            ik_solutions[7][1] = q2_2

        ik_errors = []
        valid_solutions = []
        valid_ik_errors = []
        for ik_sol in ik_solutions:
            # Calculate the forward kinematics for each solution
            fk_pos, _ = self.forward_kinematics_span_tree(ik_sol)
            # Calculate the errors for each solution
            ik_error = sqrt( (fk_pos[0]-x)**2 + (fk_pos[1]-y)**2 + (fk_pos[2]-z)**2 )
            ik_errors.append(ik_error)
            if ik_error < self.success_thresh:
                valid_solutions.append(ik_sol)
                valid_ik_errors.append(ik_error)

        if not valid_solutions:
            return None, False, None, None
        
        if len(valid_solutions) == 1 or reference_joint_pos is None:
            # If only one valid solution, return it directly
            return valid_solutions[0], True, ik_solutions, valid_ik_errors[0]
        else:
            # Calculate errors with respect to the reference joint positions
            ref_errors = []
            for sol in valid_solutions:
                ref_error = sqrt( (sol[0]-reference_joint_pos[0])**2 + (sol[1]-reference_joint_pos[1])**2 + (sol[2]-reference_joint_pos[2])**2 )
                ref_errors.append(ref_error)

            # Find the solution with the minimum error
            min_ref_error = min(ref_errors)
            best_solution_index = ref_errors.index(min_ref_error)
            best_solution = valid_solutions[best_solution_index]

            # Return the best solution, success flag, all solutions, and their errors
            return best_solution, True, ik_solutions, valid_ik_errors[best_solution_index]
    
    def wrap_q2(self, q2: float) -> float:
        """Wraps the angle to the range [0, 2*pi].
        
        The logic is:
        If angle < 0, add pi;
        if angle still < 0, add pi again. (Means originally angle was < -pi)
        """
        q2_actuator = q2 - self.thigh_offset
        if  q2_actuator < 0:
            if q2_actuator < -pi:
                # q2 < -pi
                q2_actuator += 2*pi
            else:
                # -pi <= q2 < 0
                q2_actuator += pi
        return q2_actuator + self.thigh_offset  # Add the thigh offset back to get the actuator joint position
    
    @abstractmethod
    def forward_kinematics_span_tree(self, span_tree_joint_pos: list) -> list:
        """ Forward kinematics for the span tree joint positions.
        Args:
            span_tree_joint_pos: Span tree joint positions.
        Returns:
            list: Forward kinematics positions.
        """
        raise NotImplementedError("This method should be implemented in the subclass.")