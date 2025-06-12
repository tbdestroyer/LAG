import numpy as np
import math
from qm import QuineMcCluskey
from condense_ex import Explainer

class PredicateTemplate:
    def __init__(self, num_feats):
        self.num_feats = num_feats
        self.attr_names = None
    
    def predicate_set(self):
        raise NotImplementedError

    def feat_groups(self):
        raise NotImplementedError

    def translate_state(self, state):
        raise NotImplementedError
    
    def num_predicates(self):
        raise NotImplementedError
    
    def state_to_binary(self, state):
        raise NotImplementedError
    
    def reduce_logic(self, c_binary):
        #Sections of code adapted from https://gitlab.tue.nl/ha800-hri/hayes-shah/-/blob/master/hayes_shah/hs.py
        """
        c_binary: The set of abstract classes. An array of length num_abstract_classes
            c_binary[k] is an array which contains all states of the kth class
            that array contains tuples of the form (binary_state, action)
        Simplify the predicates of an abstract state into the minimal form
        This algorithm from Hayes and Shah 2017 requires the predicate features from all binary states in the target
        class (the abstract state, in this case), and the non-target class (all other abstract states)
        The target class and non-target classes should be mutually exclusive for the algorithm
        This is not reasonable to assume, given that binary states that are the same could be in different classes
            due to stochasticity in the policy and inaccuracies in the predicates
        So, the initial solution would be to loop through the other abstract states and remove the predicate states
            which are the exact same. The resulting explanations may not be completely faithful, but higher accuracy would
            come with better predicates/
        There could be an issue where no reliable predicate explanation is found, since there are just so many states in the
            other classes. In that case, I would come up with my own way of simplifying the predicates
        """

        qm = QuineMcCluskey()

        use_qm = True

        predicates = self.predicate_set()
        explanations = []
        condensed_sets = []

        for i, abs_class in enumerate(c_binary):
            target_states = [t[0] for t in abs_class]
            non_target_states = []
            for j in range(len(c_binary)):
                if j != i:
                    for k in range(len(c_binary[j])):
                        s = c_binary[j][k][0]
                        non_target_states.append(s)
            

            for s in target_states: #Loop through sets to ensure the intersection is empty
                for j, non_s in enumerate(non_target_states):
                    if np.array_equal(s, non_s):
                        non_target_states.pop(j)
            
            if use_qm:
                target_state_strings = []
                non_target_state_strings = []
        
                for s in target_states:
                    string = ''
                    for f in s:
                        string = string + str(f)
                    target_state_strings.append(string)
        
                
                for s in non_target_states:
                    string = ''
                    for f in s:
                        string = string + str(f)
                    non_target_state_strings.append(string)
                    
                
                n = len(target_states[0])
                all_bin = [bin(x)[2:].rjust(n, '0') for x in range(2**n)]
                not_valid = list(set(all_bin) - set(target_state_strings) - set(non_target_state_strings)) #All states which never appear

                
                
                minterms = qm.simplify_los(target_state_strings, not_valid)
                #print('{}: {}'.format(i+1,minterms))
                clauses = self.minterm_to_clause(minterms, predicates)
                #print('{}: {}'.format(i+1,clauses))
                explanations.append(' or '.join(clauses))
            
            else:
                proportions = np.zeros(self.num_predicates())
                for s in target_states:
                    proportions = proportions + np.array(s)
                pos_proportions = proportions / len(target_states)
                pos_explans = []
                neg_explans = []
                for j, p in enumerate(pos_proportions):
                    predicate = predicates[j]
                    if p >= 0.9:
                        pos_explans.append(predicate['true'])
                    elif p <= 0.1:
                        neg_explans.append(predicate['false'])
                
                if pos_explans == []:
                    most_common = np.argmax(pos_proportions) #Add most common occurence
                    neg_explans.append(predicates[most_common]['true'])
                    explanations.append(' and '.join(neg_explans)) #Only use neg explans if no pos exist
                else:
                    explanations.append(' and '.join(pos_explans))
                condensed_sets.append(pos_proportions)

        return explanations

    def minterm_to_clause(self, minterms, predicates):
        

        clauses = []

        for min_term in minterms:
            str_terms = []
            for i in range(len(min_term)):
                predicate = predicates[i]
                if min_term[i] == '0':
                    str_terms.append(predicate['false'])
                elif min_term[i] == '1':
                    str_terms.append(predicate['true'])

            clauses.append(' and '.join(str_terms))

        return clauses
    

    def my_translation_algo(self, c_binary):
        predicates = self.predicate_set()
        explanations = []
        condensed_sets = []

    
        for i, abs_class in enumerate(c_binary):
            target_states = [t[0] for t in abs_class]
            e = Explainer(target_states, self.feat_groups(), len(predicates), predicates)
            ex = e.full_translate()
            explanations.append(ex)
        
        return explanations
    
class PlanePredicates(PredicateTemplate):
    def __init__(self, num_feats):
        super().__init__(num_feats)
       # self.attr_names = ['Car Position', 'Car Velocity']
        self.attr_names = [ "Ego Altitude (5km)",           # [0]
    "Ego Roll Sin",                 # [1]
    "Ego Roll Cos",                 # [2]
    "Ego Pitch Sin",                # [3]
    "Ego Pitch Cos",                # [4]
    "Ego V_body_x (mh)",            # [5]
    "Ego V_body_y (mh)",            # [6]
    "Ego V_body_z (mh)",            # [7]
    "Ego Vc (mh)",                  # [8]
    "Delta V_body_x (mh)",          # [9]  (relative enemy info)
    "Delta Altitude (km)",          # [10]
    "Ego AO (rad)",                 # [11] [0, pi]
    "Ego TA (rad)",                 # [12] [0, pi]
    "Relative Distance (10km)",     # [13]
    "Side Flag",                    # [14]
    "Missile Delta V_body_x",       # [15] (relative missile info)
    "Missile Delta Altitude",       # [16]
    "Missile Ego AO",               # [17]
    "Missile Ego TA",               # [18]
    "Missile Relative Distance",    # [19]
    "Missile Side Flag"             # [20]
]
        '''
        self.language_set = np.array(['At the bottom',
                    'On the left slope',
                    'On the right slope',
                    'High up on the left slope',
                    'High up on the right slope',
                    'Moving left slowly',
                    'Moving right slowly',
                    'Not moving',
                    'Moving left quickly',
                    'Moving right quickly'])
        '''
        self.language_set = np.array([
    # Ego plane state
    "Flying at high altitude",
    "Flying at low altitude",
    "Banked left",
    "Banked right",
    "Level roll",
    "Nose up",
    "Nose down",
    "Level pitch",
    "High forward speed",
    "Low forward speed",
    "Climbing",
    "Descending",
    "High closure rate",
    "Low closure rate",

    # Relative enemy info
    "Enemy ahead",
    "Enemy behind",
    "Enemy above",
    "Enemy below",
    "Enemy to the left",
    "Enemy to the right",
    "Enemy very close",
    "Enemy far away",
    "Enemy on same altitude",
    "Enemy at higher energy state",
    "Enemy at lower energy state",
    "Enemy on left side",
    "Enemy on right side",
    "Enemy within missile range",
    "Enemy outside missile range",
    "High angle-off to enemy",
    "Low angle-off to enemy",
    "High tail aspect",
    "High nose aspect",

    # Missile info (relative)
    "Missile approaching from ahead",
    "Missile approaching from behind",
    "Missile above",
    "Missile below",
    "Missile to the left",
    "Missile to the right",
    "Missile very close",
    "Missile far away",
    "Missile on same altitude",
    "Missile within lethal range",
    "Missile outside lethal range",
    "High angle-off to missile",
    "Low angle-off to missile",
    "Missile on left side",
    "Missile on right side",
])
    def predicate_set(self):
        '''
        predicates = [{'true': 'At the bottom', 'false': 'Not at the bottom'},
                      {'true': 'On the left slope', 'false': 'Not on the left slope'},
                      {'true': 'On the right slope', 'false': 'Not on the right slope'},
                      {'true': 'High up on the left slope', 'false': 'Not high up on the left slope'},
                      {'true': 'High up on the right slope', 'false': 'Not high up on the right slope'},
                      {'true': 'Moving left slowly', 'false': 'Not moving left slowly'},
                      {'true': 'Moving right slowly', 'false': 'Not moving right slowly'},
                      {'true': 'Not moving', 'false': 'Not moving'},
                      {'true': 'Moving left quickly', 'false': 'Not moving left quickly'},
                      {'true': 'Moving right quickly', 'false': 'Not moving right quickly'}]
                      '''
        predicates = [
    {'true': 'Flying at high altitude', 'false': 'Not flying at high altitude'},
    {'true': 'Flying at low altitude', 'false': 'Not flying at low altitude'},
    {'true': 'Banked left', 'false': 'Not banked left'},
    {'true': 'Banked right', 'false': 'Not banked right'},
    {'true': 'Level roll', 'false': 'Not level roll'},
    {'true': 'Nose up', 'false': 'Not nose up'},
    {'true': 'Nose down', 'false': 'Not nose down'},
    {'true': 'Level pitch', 'false': 'Not level pitch'},
    {'true': 'High forward speed', 'false': 'Not high forward speed'},
    {'true': 'Low forward speed', 'false': 'Not low forward speed'},
    {'true': 'Climbing', 'false': 'Not climbing'},
    {'true': 'Descending', 'false': 'Not descending'},
    {'true': 'High closure rate', 'false': 'Not high closure rate'},
    {'true': 'Low closure rate', 'false': 'Not low closure rate'},
    {'true': 'Enemy ahead', 'false': 'Enemy not ahead'},
    {'true': 'Enemy behind', 'false': 'Enemy not behind'},
    {'true': 'Enemy above', 'false': 'Enemy not above'},
    {'true': 'Enemy below', 'false': 'Enemy not below'},
    {'true': 'Enemy to the left', 'false': 'Enemy not to the left'},
    {'true': 'Enemy to the right', 'false': 'Enemy not to the right'},
    {'true': 'Enemy very close', 'false': 'Enemy not very close'},
    {'true': 'Enemy far away', 'false': 'Enemy not far away'},
    {'true': 'Enemy on same altitude', 'false': 'Enemy not on same altitude'},
    {'true': 'Enemy at higher energy state', 'false': 'Enemy not at higher energy state'},
    {'true': 'Enemy at lower energy state', 'false': 'Enemy not at lower energy state'},
    {'true': 'Enemy on left side', 'false': 'Enemy not on left side'},
    {'true': 'Enemy on right side', 'false': 'Enemy not on right side'},
    {'true': 'Enemy within missile range', 'false': 'Enemy not within missile range'},
    {'true': 'Enemy outside missile range', 'false': 'Enemy not outside missile range'},
    {'true': 'High angle-off to enemy', 'false': 'Not high angle-off to enemy'},
    {'true': 'Low angle-off to enemy', 'false': 'Not low angle-off to enemy'},
    {'true': 'High tail aspect', 'false': 'Not high tail aspect'},
    {'true': 'High nose aspect', 'false': 'Not high nose aspect'},
    {'true': 'Missile approaching from ahead', 'false': 'Missile not approaching from ahead'},
    {'true': 'Missile approaching from behind', 'false': 'Missile not approaching from behind'},
    {'true': 'Missile above', 'false': 'Missile not above'},
    {'true': 'Missile below', 'false': 'Missile not below'},
    {'true': 'Missile to the left', 'false': 'Missile not to the left'},
    {'true': 'Missile to the right', 'false': 'Missile not to the right'},
    {'true': 'Missile very close', 'false': 'Missile not very close'},
    {'true': 'Missile far away', 'false': 'Missile not far away'},
    {'true': 'Missile on same altitude', 'false': 'Missile not on same altitude'},
    {'true': 'Missile within lethal range', 'false': 'Missile not within lethal range'},
    {'true': 'Missile outside lethal range', 'false': 'Missile not outside lethal range'},
    {'true': 'High angle-off to missile', 'false': 'Not high angle-off to missile'},
    {'true': 'Low angle-off to missile', 'false': 'Not low angle-off to missile'},
    {'true': 'Missile on left side', 'false': 'Missile not on left side'},
    {'true': 'Missile on right side', 'false': 'Missile not on right side'},
]
        return predicates

    def state_to_binary(self, state):
        '''
        binary_set = [self.at_bottom(state),
                      self.on_left_slope(state),
                      self.on_right_slope(state),
                      self.high_on_left(state),
                      self.high_on_right(state),
                      self.moving_left_slow(state),
                      self.moving_right_slow(state),
                      self.not_moving(state),
                      self.moving_left_fast(state),
                      self.moving_right_fast(state)]
        '''
    
        binary_set = [
            self.high_altitude(state),
            self.low_altitude(state),
            self.banked_left(state),
            self.banked_right(state),
            self.level_roll(state),
            self.nose_up(state),
            self.nose_down(state),
            self.level_pitch(state),
            self.high_forward_speed(state),
            self.low_forward_speed(state),
            self.climbing(state),
            self.descending(state),
            self.high_closure_rate(state),
            self.low_closure_rate(state),
            self.enemy_ahead(state),
            self.enemy_behind(state),
            self.enemy_above(state),
            self.enemy_below(state),
            self.enemy_to_left(state),
            self.enemy_to_right(state),
            self.enemy_very_close(state),
            self.enemy_far_away(state),
            self.enemy_same_altitude(state),
            self.enemy_higher_energy(state),
            self.enemy_lower_energy(state),
            self.enemy_on_left(state),
            self.enemy_on_right(state),
            self.enemy_within_missile_range(state),
            self.enemy_outside_missile_range(state),
            self.high_angle_off_enemy(state),
            self.low_angle_off_enemy(state),
            self.high_tail_aspect(state),
            self.high_nose_aspect(state),
            self.missile_approaching_front(state),
            self.missile_approaching_behind(state),
            self.missile_above(state),
            self.missile_below(state),
            self.missile_to_left(state),
            self.missile_to_right(state),
            self.missile_very_close(state),
            self.missile_far_away(state),
            self.missile_same_altitude(state),
            self.missile_within_lethal_range(state),
            self.missile_outside_lethal_range(state),
            self.high_angle_off_missile(state),
            self.low_angle_off_missile(state),
            self.missile_on_left(state),
            self.missile_on_right(state),
]
        return np.array(binary_set)

    
    def translate_state(self, binary_set):
        '''
        language_set = np.array(['At the bottom',
                    'On the left slope',
                    'On the right slope',
                    'High up on the left slope',
                    'High up on the right slope',
                    'Moving left slow',
                    'Moving right slow',
                    'Not moving',
                    'Moving left fast',
                    'Moving right fast'])
        '''
        language_set = np.array([
    "Flying at high altitude",
    "Flying at low altitude",
    "Banked left",
    "Banked right",
    "Nose up",
    "Nose down",
    "High forward speed",
    "Low forward speed",
    "Enemy ahead",
    "Enemy very close"
])
        
        idx = np.where(binary_set==1)[0]
        true_set = language_set[idx]
        string = ''
        if true_set.size != 0:
            string = true_set[0]
            if true_set[1:].size != 0:
                for pred in true_set[1:]:
                    string = string + ' and '
                    string = string + pred
        
        return string

    def feat_groups(self):
        #groups = [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]
        groups = [
        [0, 1, 2, 3, 4],           # Ego plane orientation/attitude (altitude, roll, pitch)
        [5, 6, 7, 8],              # Ego plane velocities (v_body_x, v_body_y, v_body_z, Vc)
        [9, 10, 11, 12, 13, 14],   # Relative enemy info (delta_v, delta_alt, AO, TA, distance, side_flag)
        [15, 16, 17, 18, 19, 20]   # Relative missile info (same structure as enemy)
    ]
        return groups

    def num_predicates(self):
        return 48
    '''
    def at_bottom(self, state):
        if state[0] >= -0.6 and state[0] <= -0.4:
            return 1
        else:
            return 0
    
    def on_left_slope(self, state):
        if state[0] < -0.6 and state[0] > -0.9:
            return 1
        else:
            return 0
    
    def on_right_slope(self, state):
        if state[0] > -0.4 and state[0] < 0.3:
            return 1
        else:
            return 0
    
    def high_on_left(self, state):
        if state[0] <= -0.9:
            return 1
        else:
            return 0
    
    def high_on_right(self, state):
        if state[0] >= 0.3:
            return 1
        else:
            return 0
    
    def moving_left_slow(self, state):
        if state[1] < 0 and state[1] > -0.025:
            return 1
        else:
            return 0
    
    def moving_right_slow(self, state):
        if state[1] > 0 and state[1] < 0.025:
            return 1
        else:
            return 0
    
    def not_moving(self, state):
        if state[1] == 0:
            return 1
        else:
            return 0
    
    def moving_left_fast(self, state):
        if state[1] <= -0.025:
            return 1
        else:
            return 0
    
    def moving_right_fast(self, state):
        if state[1] >= 0.025:
            return 1
        else:
            return 0
'''
    def high_altitude(self, state):
    # Example: state[0] is altitude in 5km units
        return 1 if state[0] > 2.0 else 0  # >10km

    def low_altitude(self, state):
        return 1 if state[0] < 1.0 else 0  # <5km

    def banked_left(self, state):
        # state[1] is roll_sin, negative means left bank
        return 1 if state[1] < -0.5 else 0

    def banked_right(self, state):
        # state[2] is roll_cos, positive means right bank
        return 1 if state[2] > 0.5 else 0

    def nose_up(self, state):
        # state[3] is pitch_sin, positive means nose up
        return 1 if state[3] > 0.5 else 0

    def nose_down(self, state):
        # state[4] is pitch_cos, negative means nose down
        return 1 if state[4] < -0.5 else 0

    def high_forward_speed(self, state):
        # state[5] is v_body_x (forward speed, in mh)
        return 1 if state[5] > 0.8 else 0

    def low_forward_speed(self, state):
        return 1 if state[5] < 0.3 else 0

    def enemy_ahead(self, state):
        # state[12] is ego_TA (target aspect), small means ahead
        return 1 if state[12] < (np.pi / 4) else 0

    def enemy_very_close(self, state):
        # state[13] is relative distance (in 10km units)
        return 1 if state[13] < 1.0 else 0  # <10km
    
    def level_roll(self, state):
        return 1 if np.abs(state[1]) < 0.1 else 0

    def level_pitch(self, state):
        return 1 if np.abs(state[3]) < 0.1 else 0

    def climbing(self, state):
        return 1 if state[7] > 0 else 0

    def descending(self, state):
        return 1 if state[7] < 0 else 0

    def high_closure_rate(self, state):
        return 1 if state[9] > 0.8 else 0

    def low_closure_rate(self, state):
        return 1 if state[9] < 0.2 else 0

    def enemy_behind(self, state):
        return 1 if state[12] > np.pi * 3/4 else 0

    def enemy_above(self, state):
        return 1 if state[10] < -0.5 else 0

    def enemy_below(self, state):
        return 1 if state[10] > 0.5 else 0

    def enemy_to_left(self, state):
        return 1 if state[14] == -1 else 0

    def enemy_to_right(self, state):
        return 1 if state[14] == 1 else 0

    def enemy_same_altitude(self, state):
        return 1 if np.abs(state[10]) < 0.1 else 0

    def enemy_higher_energy(self, state):
        return 1 if state[8] < 0.5 else 0

    def enemy_lower_energy(self, state):
        return 1 if state[8] > 1.5 else 0

    def enemy_on_left(self, state):
        return 1 if state[14] == -1 else 0

    def enemy_on_right(self, state):
        return 1 if state[14] == 1 else 0

    def enemy_within_missile_range(self, state):
        return 1 if state[13] < 1.5 else 0

    def enemy_outside_missile_range(self, state):
        return 1 if state[13] > 3.0 else 0

    def high_angle_off_enemy(self, state):
        return 1 if state[11] > np.pi/2 else 0

    def low_angle_off_enemy(self, state):
        return 1 if state[11] < np.pi/6 else 0

    def high_tail_aspect(self, state):
        return 1 if state[12] > 2.5 else 0

    def high_nose_aspect(self, state):
        return 1 if state[12] < 0.5 else 0

    def missile_approaching_front(self, state):
        return 1 if state[18] < 0.5 else 0

    def missile_approaching_behind(self, state):
        return 1 if state[18] > 2.5 else 0

    def missile_above(self, state):
        return 1 if state[16] < -0.5 else 0

    def missile_below(self, state):
        return 1 if state[16] > 0.5 else 0

    def missile_to_left(self, state):
        return 1 if state[20] == -1 else 0

    def missile_to_right(self, state):
        return 1 if state[20] == 1 else 0

    def missile_very_close(self, state):
        return 1 if state[19] < 1.0 else 0

    def missile_far_away(self, state):
        return 1 if state[19] > 3.0 else 0

    def missile_same_altitude(self, state):
        return 1 if np.abs(state[16]) < 0.1 else 0

    def missile_within_lethal_range(self, state):
        return 1 if state[19] < 2.0 else 0

    def missile_outside_lethal_range(self, state):
        return 1 if state[19] > 4.0 else 0

    def high_angle_off_missile(self, state):
        return 1 if state[17] > np.pi/2 else 0

    def low_angle_off_missile(self, state):
        return 1 if state[17] < np.pi/6 else 0

    def missile_on_left(self, state):
        return 1 if state[20] == -1 else 0

    def missile_on_right(self, state):
        return 1 if state[20] == 1 else 0
    
    def enemy_far_away(self,state):
        return 1 if state[13] > 3.0 else 0
    
    
class LunarLanderPredicates(PredicateTemplate):
    def __init__(self, num_feats):
        super().__init__(num_feats)
        self.attr_names = ['X Coordinate',
                           'Y Coordinate',
                           'X Velocity',
                           'Y Velocity',
                           'Lander Angle',
                           'Angular Velocity',
                           'Left leg on ground',
                           'Right leg on ground']
        self.language_set = np.array(['Left of the goal', 'Right of the goal',
                                      'On top of goal', 'Higher than goal', 'Same height as goal',
                                      'Left leg on ground', 'Right leg on ground',
                                      'Lander tilted left', 'Lander tilted right',
                                      'Moving right', 'Moving left'])
    
    def state_to_binary(self, state):
        b = [self.left_of_goal(state),
             self.right_of_goal(state),
             self.on_top_of_goal(state),
             self.higher_than_goal(state),
             self.at_same_height(state),
             self.left_leg_on_ground(state),
             self.right_leg_on_ground(state),
             self.tilted_left(state),
             self.tilted_right(state),
             self.moving_right(state),
             self.moving_left(state)]
        return np.array(b)
    
    def translate_state(self, binary_set):
        idx = np.where(binary_set==1)[0]
        true_set = self.language_set[idx]
        string = ''
        if true_set.size != 0:
            string = true_set[0]
            if true_set[1:].size != 0:
                for pred in true_set[1:]:
                    string = string + ' and '
                    string = string + pred
        
        return string

    def feat_groups(self):
        groups = [[0, 1, 2], [3, 4], [5], [6], [7, 8], [9, 10]]
        return groups
    
    def predicate_set(self):
        predicates = [{'true': 'Left of the goal', 'false': 'Not left of the goal'},
                      {'true': 'Right of the goal', 'false': 'Not right of the goal'},
                      {'true': 'Directly on top of goal', 'false': 'Not directly on top of goal'},
                      {'true': 'Higher than goal', 'false': 'Not higher than goal'},
                      {'true': 'At same height as goal', 'false': 'Not at same height as goal'},
                      {'true': 'Left leg on the ground', 'false': 'Left leg not on the ground'},
                      {'true': 'Right leg on the ground', 'false': 'Right leg not on the ground'},
                      {'true': 'Lander tilted left', 'false': 'Lander not tilted left'},
                      {'true': 'Lander tilted right', 'false': 'Lander not tilted right'},
                      {'true': 'Moving right', 'false': 'Not moving right'},
                      {'true': 'Moving left', 'false': 'Not moving left'}]
        return predicates
    
    def num_predicates(self):
        return len(self.predicate_set())
    
    def left_of_goal(self, state):
        if state[0] < -0.08:
            return 1
        else:
            return 0
    
    def right_of_goal(self, state):
        if state[0] > 0.08:
            return 1
        else:
            return 0
    
    def on_top_of_goal(self, state):
        if np.abs(state[0]) <= 0.08:
            return 1
        else:
            return 0
    
    def higher_than_goal(self, state):
        if state[1] > 0.08:
            return 1
        else:
            return 0
    
    def at_same_height(self, state):
        if state[1] <= 0.08:
            return 1
        else:
            return 0
    
    def left_leg_on_ground(self, state):
        if state[6] == 1:
            return 1
        else:
            return 0
    
    def right_leg_on_ground(self, state):
        if state[7] == 1:
            return 1
        else:
            return 0
    
    def tilted_left(self, state):
        if state[4] < -0.3:
            return 1
        else:
            return 0
    def tilted_right(self, state):
        if state[4] > 0.3:
            return 1
        else:
            return 0
    
    def moving_right(self, state):
        if state[2] > 0.01:
            return 1
        else:
            return 0
    
    def moving_left(self, state):
        if state[2] < -0.01:
            return 1
        else:
            return 0
    
    



class BlackjackPredicates(PredicateTemplate):
    def __init__(self, num_feats):
        super().__init__(num_feats)
        self.attr_names = ['Current sum', 'Dealer card', 'Usable ace']
        self.language_set = np.array(['sum less than 14', 'sum 14-16','sum 17-19',
                                      'sum 20-21', ' d sum less 7', 'd sum 7-9',
                                      'd sum 10-ace','ace 11'])
    

    def predicate_set(self):
        predicates = [{'true': 'Sum less than 14', 'false': 'Sum not less than 14'},
                      {'true': 'Sum of 14-16', 'false': 'Sum not of 14-16'},
                      {'true': 'Sum of 17-19', 'false': 'Sum not of 17-19'},
                      {'true': 'Sum of 20-21', 'false': 'Sum not of 20-21'},
                      {'true': 'Dealer card less than 7', 'false': 'Dealer card 7 or more'},
                      {'true': 'Dealer card 7-9', 'false': 'Dealer card not 7-9'},
                      {'true': 'Dealer card 10 or ace', 'false': 'Dealer card not 10 or ace'},
                      {'true': 'Ace is 11', 'false': 'No ace or ace is not 11'}]
        
        return predicates
    
    def num_predicates(self):
        return len(self.predicate_set())
    
    def feat_groups(self):
        groups = [[0, 1, 2, 3], [4, 5, 6], [7]]
        return groups


    def state_to_binary(self, state):
        b = [self.less_14(state),
             self.p14_16(state),
             self.p17_19(state),
             self.p20_21(state),
             self.dless_7(state),
             self.d7_9(state),
             self.d10_ace(state),
             self.use_ace(state)]
        return np.array(b)
    
    def translate_state(self, binary_set):
        idx = np.where(binary_set==1)[0]
        true_set = self.language_set[idx]
        string = ''
        if true_set.size != 0:
            string = true_set[0]
            if true_set[1:].size != 0:
                for pred in true_set[1:]:
                    string = string + ' and '
                    string = string + pred
        
        return string


    def less_14(self, state):
        if state[0] < 14:
            return 1
        else:
            return 0
    
    def p14_16(self, state):
        if state[0] >= 14 and state[0] <= 16:
            return 1
        else:
            return 0
    
    def p17_19(self, state):
        if state[0] >= 17 and state[0] <= 19:
            return 1
        else:
            return 0
    
    def p20_21(self, state):
        if state[0] == 20 or state[0] == 21:
            return 1
        else:
            return 0

    def dless_7(self, state):
        if state[1] < 7 and state[1] != 1:
            return 1
        else:
            return 0
    
    def d7_9(self, state):
        if state[1] >= 7 and state[1] <= 9:
            return 1
        else:
            return 0
    
    def d10_ace(self, state):
        if state[1] == 10 or state[1] == 1:
            return 1
        else:
            return 0
    
    
    def use_ace(self, state):
        if state[2] == 1:
            return 1
        else:
            return 0
    
class MountainCarPredicates(PredicateTemplate):
    def __init__(self, num_feats):
        super().__init__(num_feats)
        self.attr_names = ['Car Position', 'Car Velocity']
        self.language_set = np.array(['At the bottom',
                    'On the left slope',
                    'On the right slope',
                    'High up on the left slope',
                    'High up on the right slope',
                    'Moving left slowly',
                    'Moving right slowly',
                    'Not moving',
                    'Moving left quickly',
                    'Moving right quickly'])

    def predicate_set(self):
        predicates = [{'true': 'At the bottom', 'false': 'Not at the bottom'},
                      {'true': 'On the left slope', 'false': 'Not on the left slope'},
                      {'true': 'On the right slope', 'false': 'Not on the right slope'},
                      {'true': 'High up on the left slope', 'false': 'Not high up on the left slope'},
                      {'true': 'High up on the right slope', 'false': 'Not high up on the right slope'},
                      {'true': 'Moving left slowly', 'false': 'Not moving left slowly'},
                      {'true': 'Moving right slowly', 'false': 'Not moving right slowly'},
                      {'true': 'Not moving', 'false': 'Not moving'},
                      {'true': 'Moving left quickly', 'false': 'Not moving left quickly'},
                      {'true': 'Moving right quickly', 'false': 'Not moving right quickly'}]
        return predicates

    def state_to_binary(self, state):
        binary_set = [self.at_bottom(state),
                      self.on_left_slope(state),
                      self.on_right_slope(state),
                      self.high_on_left(state),
                      self.high_on_right(state),
                      self.moving_left_slow(state),
                      self.moving_right_slow(state),
                      self.not_moving(state),
                      self.moving_left_fast(state),
                      self.moving_right_fast(state)]
        
        return np.array(binary_set)
    
    def translate_state(self, binary_set):
        language_set = np.array(['At the bottom',
                    'On the left slope',
                    'On the right slope',
                    'High up on the left slope',
                    'High up on the right slope',
                    'Moving left slow',
                    'Moving right slow',
                    'Not moving',
                    'Moving left fast',
                    'Moving right fast'])
        
        idx = np.where(binary_set==1)[0]
        true_set = language_set[idx]
        string = ''
        if true_set.size != 0:
            string = true_set[0]
            if true_set[1:].size != 0:
                for pred in true_set[1:]:
                    string = string + ' and '
                    string = string + pred
        
        return string

    def feat_groups(self):
        groups = [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]
        return groups

    def num_predicates(self):
        return 10
    
    def at_bottom(self, state):
        if state[0] >= -0.6 and state[0] <= -0.4:
            return 1
        else:
            return 0
    
    def on_left_slope(self, state):
        if state[0] < -0.6 and state[0] > -0.9:
            return 1
        else:
            return 0
    
    def on_right_slope(self, state):
        if state[0] > -0.4 and state[0] < 0.3:
            return 1
        else:
            return 0
    
    def high_on_left(self, state):
        if state[0] <= -0.9:
            return 1
        else:
            return 0
    
    def high_on_right(self, state):
        if state[0] >= 0.3:
            return 1
        else:
            return 0
    
    def moving_left_slow(self, state):
        if state[1] < 0 and state[1] > -0.025:
            return 1
        else:
            return 0
    
    def moving_right_slow(self, state):
        if state[1] > 0 and state[1] < 0.025:
            return 1
        else:
            return 0
    
    def not_moving(self, state):
        if state[1] == 0:
            return 1
        else:
            return 0
    
    def moving_left_fast(self, state):
        if state[1] <= -0.025:
            return 1
        else:
            return 0
    
    def moving_right_fast(self, state):
        if state[1] >= 0.025:
            return 1
        else:
            return 0

class GridworldPredicates(PredicateTemplate):
    def __init__(self, num_feats):
        super().__init__(num_feats)
        self.attr_names = ['Position']
        self.language_set = np.array(['At the start',
                    'Reached the goal',
                    'A cliff is below',
                    'On the left border',
                    'On the right border',
                    'In free space',
                    'Near the goal'])

    def state_to_coords(self, state):
        obs = state[0]
        coords = (obs // 12, obs % 12)
        return coords

    def predicate_set(self):
        predicates = [{'true': 'At the start', 'false': 'Not at the start'},
                      {'true': 'Reached the goal', 'false': 'Has not reached the goal'},
                      {'true': 'A cliff is below', 'false': 'A cliff is not below'},
                      {'true': 'On the left border', 'false': 'Not on the left border'},
                      {'true': 'On the right border', 'false': 'Not on the right border'},
                      {'true': 'In free space', 'false': 'Not in free space'},
                      {'true': 'Near the goal', 'false': 'Not near the goal'}]
    
        return predicates
    
    def state_to_binary(self, state):
        coords = self.state_to_coords(state)
        binary_set = [self.at_start(coords),
                      self.at_goal(coords),
                      self.cliff_below(coords),
                      self.at_left_edge(coords),
                      self.at_right_edge(coords),
                      self.in_free_space(coords),
                      self.near_goal(coords)]
        
        return np.array(binary_set)
    
    def translate_state(self, binary_set):
        language_set = np.array(['At the start',
                    'Reached the goal',
                    'A cliff is below',
                    'On the left border',
                    'On the right border',
                    'In free space',
                    'Near the goal'])
        
        idx = np.where(binary_set==1)[0]
        true_set = language_set[idx]
        string = ''
        if true_set.size != 0:
            string = true_set[0]
            if true_set[1:].size != 0:
                for pred in true_set[1:]:
                    string = string + ' and '
                    string = string + pred
        
        return string


    def feat_groups(self):
        groups = [[0, 1, 2, 3, 4, 5], [6]]
        return groups

    def num_predicates(self):
        return 7

    def at_start(self, coords):
        if coords[0] == 3 and coords[1] == 0:
            return 1
        else:
            return 0
    
    def at_goal(self, coords):
        if coords[0] == 3 and coords[1] == 11:
            return 1
        else:
            return 0

    def cliff_below(self, coords):
        if coords[0] == 2 and coords[1] > 0 and coords[1] < 11:
            return 1
        else:
            return 0
    
    def at_left_edge(self, coords):
        if coords[0] < 3 and coords[1] == 0:
            return 1
        else:
            return 0
    
    def at_right_edge(self, coords):
        if coords[0] < 3 and coords[1] == 11:
            return 1
        else:
            return 0
    
    def in_free_space(self, coords):
        if coords[0] < 2 and coords[1] > 0 and coords[1] < 11:
            return 1
        else:
            return 0

    def near_goal(self, coords):
        if coords[0] > 1 and coords[1] > 9:
            return 1
        else:
            return 0




class CartpolePredicates(PredicateTemplate):
    def __init__(self, num_feats):
        super().__init__(num_feats)
        self.attr_names = ['Cart Position', 'Cart Velocity', 'Pole Angle', 'Pole Angular Velocity']
        self.language_set = np.array(['Pole is falling to the left',
                    'Pole is falling to the right',
                    'Pole is stabilizing from left',
                    'Pole is stabilizing from right',
                    'Pole is standing up',
                    'Cart is moving left',
                    'Cart is moving right',
                    'Cart is on the left',
                    'Cart is on the right',
                    'Cart is in the middle'])

    def predicate_set(self):
        predicates = [{'true': 'Pole is falling to the left', 'false': 'Pole is not falling to the left'},
                      {'true': 'Pole is falling to the right', 'false': 'Pole is not falling to the right'},
                      {'true': 'Pole is stabilizing to the left', 'false': 'Pole is not stabilizing to the left'},
                      {'true': 'Pole is stabilizing to the right', 'false': 'Pole is not stabilizing to the right'},
                      {'true': 'Pole is standing up', 'false': 'Pole is not standing up'},
                      {'true': 'Cart is moving left', 'false': 'Cart is not moving left'},
                      {'true': 'Cart is moving right', 'false': 'Cart is not moving right'},
                      {'true': 'Cart is on the left', 'false': 'Cart is not on the left'},
                      {'true': 'Cart is on the right', 'false': 'Cart is not on the right'},
                      {'true': 'Cart is in the middle', 'false': 'Cart is not in the middle'}]
    
        return predicates
    
    def translate_state(self, binary_set):
        language_set = np.array(['Pole is falling to the left',
                    'Pole is falling to the right',
                    'Pole is stabilizing from left',
                    'Pole is stabilizing from right',
                    'Pole is standing up',
                    'Cart is moving left',
                    'Cart is moving right',
                    'Cart is on the left',
                    'Cart is on the right',
                    'Cart is in the middle'])
        
        idx = np.where(binary_set==1)[0]
        true_set = language_set[idx]
        string = ''
        if true_set.size != 0:
            string = true_set[0]
            if true_set[1:].size != 0:
                for pred in true_set[1:]:
                    string = string + ' and '
                    string = string + pred
        
        return string
    
    def feat_groups(self):
        groups = [[0, 1, 2, 3, 4], [5, 6], [7, 8, 9]]
        return groups

    def num_predicates(self):
        return 10
    
    def state_to_binary(self, state):
        state = np.reshape(state, [-1])
        binary = [self.pole_fall_left(state),
                  self.pole_fall_right(state),
                  self.pole_stabilize_left(state),
                  self.pole_stabilize_right(state),
                  self.pole_standing_up(state),
                  self.cart_moving_left(state),
                  self.cart_moving_right(state),
                  self.cart_pos_left(state),
                  self.cart_pos_right(state),
                  self.cart_near_middle(state)]
        
        return np.array(binary)
    
    def pole_fall_left(self, state):
        if state[2] < -0.01 and state[3] < 0:
            return 1
        else:
            return 0
    
    def pole_fall_right(self, state):
        if state[2] > 0.01 and state[3] > 0:
            return 1
        else:
            return 0
    
    def pole_stabilize_left(self, state):
        if state[2] < -0.01 and state[3] > 0:
            return 1
        else:
            return 0
    
    def pole_stabilize_right(self, state):
        if state[2] > 0.01 and state[3] < 0:
            return 1
        else:
            return 0
    
    def pole_standing_up(self, state):
        if np.abs(state[2]) <= 0.01:
            return 1
        else:
            return 0
    
    def cart_moving_left(self, state):
        if state[1] < 0:
            return 1
        else:
            return 0
    
    def cart_moving_right(self, state):
        if state[1] >= 0:
            return 1
        else:
            return 0
    
    def cart_pos_left(self, state):
        if state[0] < -0.05:
            return 1
        else:
            return 0
    
    def cart_pos_right(self, state):
        if state[0] > 0.05:
            return 1
        else:
            return 0
    
    def cart_near_middle(self, state):
        if np.abs(state[0]) < 0.05:
            return 1
        else:
            return 0

class AcornPredicates(PredicateTemplate):
    def __init__(self, num_feats, num_bins=10):
        super().__init__(num_feats)
        self.num_bins = num_bins
        self.predicate_set = []
        self.binned_indices = []

        self.agent1_features = ['Percent Tweets', 'Percent Replies',
                                'Percent Retweets', 'Percent Mentions',
                                'Percent Followers']

    
    def translate_state(self, state): #Temporary implementation. Later, will include other ways of predicate grounding
        self.predicate_set = []
        binned = self.binning(state)
        for i, feature in enumerate(binned):
            self.binned_indices.append(len(self.predicate_set))
            self.predicate_set.append(feature)
        
        nl_predicate_set = self.nl_grounding()

        return self.predicate_set, nl_predicate_set

    def state_to_binary(self, state):
        return self.binning(state)
    
    def binning(self, state):
        binned_state = []
        for feature in state:
            assert 0 <= feature and feature <= 1

            idx = math.floor(feature * self.num_bins)
            binary = np.zeros(self.num_bins)
            binary[idx] = 1
            binned_state.append(binary)
        binned_state = np.reshape(np.array(binned_state), [-1])
    
        return binned_state
    
    def nl_grounding(self): #Temporary. Later, will include translations for other types of predicates
        for i, pred in enumerate(self.predicate_set):
            if i in self.binned_indices:
                feat_idx = np.where(np.array(self.binned_indices==i))[0][0]
                self.nl_predicate_set.append(self.translate_bins(pred, feat_idx))
        
        return self.nl_predicate_set
        
        

    
    def translate_bins(self, predicate, feat_idx):
        idx = np.argmax(predicate)
        low_bound = idx / self.num_bins
        high_bound = low_bound + (1 / self.num_bins)

        string = "{} is between {} and {}".format(self.agent1_features[feat_idx], low_bound, high_bound)
        return string

    def num_predicates(self):
        return self.num_feats * self.num_bins



