class NaturalConstraint:
    @staticmethod
    def constant_length_square(first_vector, second_vector, first_length, second_length):
        return first_vector * second_vector - first_length * second_length

    @staticmethod
    def constant_angle(first_vector, second_vector, length, angle):
        return first_vector * second_vector - length * np.cos(angle)
