from distutils.core import setup

setup( name='learning_table_tennis_from_scratch',
       version='1.0',
       description='hysr',
       classifiers=[
           'License :: OSI Approved :: MIT License',
           'Programming Language :: Python :: 3',
       ],
       keywords='hysr reinforcement_learning table_tennis',
       url='https://github.com/intelligent-soft-robots/learning_table_tennis_from_scratch',
       author='Vincent Berenz',
       author_email='vberenz@tuebingen.mpg.de',
       license='MIT',
       packages=['learning_table_tennis_from_scratch'],
       install_requires=['gym'],
       zip_safe=True,
       data_files=[('learning_table_tennis_from_scratch_config/',
                    ["config/hysr_one_ball_default.json",
                     "config/ppo_default.json",
                     "config/reward_default.json"])],
       scripts=['bin/hysr_one_ball_random',
                'bin/hysr_one_ball_swing',
                'bin/hysr_one_ball_ppo',
                'bin/hysr_one_ball_reset',
                'bin/hysr_one_ball_reward_tests',
                "bin/hysr_start_robots",
                "bin/hysr_start_robots_accelerated"]
)



