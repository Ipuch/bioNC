from .three_link_pendulum import build_n_link_pendulum

model = build_n_link_pendulum(nb_segments=3)

# Options 1
# external_force_set = ExternalForceSet.empty_from_nb_segment(model.nb_segments)

# Options 2
external_force_set = model.external_force_set()

# add forces and moment
# point_of_application = [0, 0, 0]
# moment_only = [0, 0, 0]
# moment_and_force = [1, 0, 0, 0, 0, 0]

external_force_set.add_in_global("segment", moment_and_force, point_of_application)
external_force_set.add_in_global_local_point("segment", moment_and_force, point_of_application)
external_force_set.add_in_local("segment", moment_and_force, point_of_application)





