import unittest,tempfile
from learning_table_tennis_from_scratch.rewards import *


class REWARDS_TESTCASE(unittest.TestCase):

    def setUp(self):
        pass

    
    def tearDown(self):
        pass

    
    def _get_json_file(self,
                         normalization_constant,
                         rtt_cap,
                         smash):
        f = tempfile.NamedTemporaryFile()
        fields = ",\n\t".join(['"normalization_constance":{}',
                               '"rtt_cap:{}"',
                               '"smash:{}"']).format(normalization_constant,
                                                     rtt_cap,
                                                     smash)
        f.write('{\n\t{{}\n}'.format(fields))
        return f
        
    
    def test_read_default_json(self):
        '''
        testing reading the default config file
        does not thrown an exception
        '''
        jsonpath = JsonReward.default_path()
        no_exception = True
        try :
            reward_function = JsonReward.get(jsonpath)
        except:
            no_exception = False
        self.assertTrue(no_exception)

        
    def test_json_parsing(self):
        '''
        testing a json file results in correct
        config, including the instantion of
        a Reward
        '''
        f = self._get_json_file(0.2,-0.3,False)
        reward_fn = JsonReward.get(f)
        f.close()
        assertEqual(reward_fn.__class__,Reward)
        assertEqual(reward_fn.config.normalization_constant,0.2)
        assertEqual(reward_fn.config.rtt_cap,-0.3)

        
    def test_json_parsing_smash(self):
        '''
        testing a json file results in correct
        config, including the instantion of
        a SmashReward
        '''
        f = self._get_json_file(1,2,True)
        reward_fn = JsonReward.get(f)
        f.close()
        assertEqual(reward_fn.__class__,SmashReward)
        assertEqual(reward_fn.config.normalization_constant,1.0)
        assertEqual(reward_fn.config.rtt_cap,2.0)
        
