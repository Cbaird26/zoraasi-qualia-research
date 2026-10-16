"""Prospectively documented software-only connectivity checks."""
import json
import math
import unittest
from zora_lab.connection_study import (ConnectionObservation, STUDY_STATUS,
    telemetry, generate_trials,fit_transition,run,score)


class ConnectionStudyTests(unittest.TestCase):
    def test_telemetry_does_not_invent_feelings(self):
        for arrived in (False,True):
            for notice in (False,True):
                x=telemetry(ConnectionObservation(arrived,notice))
                self.assertIsNone(x['model_emotion'])
                self.assertIsNone(x['ai_subjective_experience'])
                self.assertIsNone(x['human_self_report'])
                self.assertEqual(x['status'],STUDY_STATUS)
                self.assertNotIn('Sad',json.dumps(x))

    def test_sham_and_actual_gap_not_conflated(self):
        announced=telemetry(ConnectionObservation(False,True,'announced_pause'))
        sham=telemetry(ConnectionObservation(True,True,'sham_notice'))
        self.assertEqual(announced['telemetry_state'],'expected_gap')
        self.assertEqual(sham['telemetry_state'],'arriving_despite_notice')
        self.assertEqual(telemetry(ConnectionObservation(False))['telemetry_state'],'unexplained_gap')

    def test_strict_types_and_conditions(self):
        for bad in (0,1,'false',None):
            with self.assertRaises(ValueError):ConnectionObservation(bad)
        with self.assertRaises(ValueError):ConnectionObservation(True,True,'unknown')
        with self.assertRaises(TypeError):telemetry({'packet_arrived':True})

    def test_repeatable_and_train_only_fit(self):
        a,b=generate_trials()
        self.assertEqual((a,b),generate_trials())
        self.assertEqual(len(a),24);self.assertEqual(len(b),12)
        fit=fit_transition(a)
        changed=[s[:] for s in b];changed[0]=[not x for x in changed[0]]
        self.assertEqual(fit,fit_transition(a))
        self.assertNotEqual(b,changed)
        self.assertTrue(all(0<v<1 for v in fit[0].values()))

    def test_no_gold_input_or_subjective_fields(self):
        r=run();self.assertEqual(r,run())
        self.assertEqual(r['evaluation_transitions'],12*29)
        self.assertTrue(r['shuffled_index_derangement'])
        self.assertTrue(r['no_raw_human_data'])
        self.assertNotIn('felt',json.dumps(r))
        self.assertIn('AI qualia',r['claims_not_supported'])
        for arm in r['arms'].values():
            self.assertTrue(all(math.isfinite(v) and v>=0 for v in arm.values()))

    def test_proper_scoring(self):
        self.assertLess(score(.9,True)['log_loss'],score(.1,True)['log_loss'])
        self.assertLess(score(.1,False)['brier'],score(.9,False)['brier'])

if __name__=='__main__': unittest.main()
