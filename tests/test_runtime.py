import json
import math
from pathlib import Path
import numpy as np
import pytest
from atlas_drive_runtime import EnvelopeProfile, evaluate

ROOT=Path(__file__).resolve().parents[1]
@pytest.fixture
def profile(): return EnvelopeProfile.load(ROOT/'examples/synthetic-envelope.json')

def test_signed_limits_and_requested_torque_are_distinct(profile):
    result=evaluate(profile,[25.,-25.,12.],[10.,-10.,0.])
    np.testing.assert_allclose(result['torque_Nm'],[20.,-20.,12.])
    assert result['saturated'].tolist()==[True,True,False]
    assert not result['fault'].any()

def test_interpolation_and_hotter_input_derating(profile):
    # Halfway between 0/10 rad/s and 25/80 C: four corners 30,20,18,12.
    r=evaluate(profile,100.,5.,52.5)
    assert r['torque_Nm']==pytest.approx(20.)
    assert evaluate(profile,10.,0.,110.)['torque_Nm']==0
    assert not evaluate(profile,0.,20.,110.)['fault']

@pytest.mark.parametrize('demand,speed,temp',[(0,21,25),(0,-21,25),(1,0,24),(1,0,111),(math.nan,0,25),(1,math.inf,25),(1,0,math.nan)])
def test_invalid_state_is_fault_even_for_zero_request(profile,demand,speed,temp):
    r=evaluate(profile,demand,speed,temp)
    assert r['fault'] and r['saturated'] and r['torque_Nm']==0

@pytest.mark.parametrize('mutate',[
    lambda d:d.update(speed_rad_s=[-20,0,0,10,20]),
    lambda d:d.update(temperature_C=[25,80]),
    lambda d:d.update(response_time_s=-1),
    lambda d:d.update(extra='unexpected private data'),
    lambda d:d['upper_torque_Nm'][0].__setitem__(0,-1),
])
def test_profile_rejects_invalid_tables_and_unknown_fields(profile,mutate):
    d=profile.document;mutate(d)
    with pytest.raises(ValueError):EnvelopeProfile(d)

def test_fingerprint_and_owned_arrays(profile,tmp_path):
    p=tmp_path/'profile.json';p.write_text(json.dumps(profile.document))
    assert EnvelopeProfile.load(p).sha256==profile.sha256
    copy=profile.document;copy['upper_torque_Nm'][0][0]=999
    assert profile.upper[0,0]==20
    with pytest.raises(ValueError):profile.upper[0,0]=999

def test_numpy_torch_parity_batched_and_edge_values(profile):
    torch=pytest.importorskip('torch')
    from atlas_drive_runtime.torch_backend import TorchEnvelope
    rng=np.random.default_rng(671)
    request=rng.uniform(-50,50,(64,12));speed=rng.uniform(-24,24,(64,12));temp=rng.uniform(20,120,(64,12))
    cpu=evaluate(profile,request,speed,temp)
    tensor=TorchEnvelope(profile,dtype=torch.float64).evaluate(request,speed,temp)
    for k,v in cpu.items():np.testing.assert_allclose(tensor[k].numpy(),v,rtol=1e-12,atol=1e-12)

def test_response_lag_shrinking_envelope_fault_latch_and_selective_reset(profile):
    torch=pytest.importorskip('torch')
    from atlas_drive_runtime.torch_backend import TorchEnvelope,TorqueState
    state=TorqueState(TorchEnvelope(profile,dtype=torch.float64),(2,1))
    r=state.step([[25.],[25.]],[[0.],[0.]],25.,.01)
    np.testing.assert_allclose(r['torque_Nm'],25*(1-math.exp(-1)))
    r=state.step([[25.],[25.]],[[0.],[0.]],110.,.01)
    assert not r['torque_Nm'].any()  # new zero envelope overrides response lag
    r=state.step([[25.],[25.]],[[21.],[0.]],25.,.01)
    assert r['fault'].tolist()==[[True],[False]]
    r=state.step([[25.],[25.]],[[0.],[0.]],25.,.01)
    assert r['torque_Nm'][0]==0 and r['fault'][0]
    held=state.torque[1].clone();state.reset(torch.tensor([0]))
    assert not state.fault.any() and torch.equal(state.torque[1],held)
    r=state.step([[25.],[25.]],[[0.],[0.]],25.,.01)
    assert r['torque_Nm'][0]>0
    with pytest.raises(ValueError):state.step([[1.]],[[0.]],25.,.01)
    with pytest.raises(ValueError):state.step([[1.],[1.]],[[0.],[0.]],25.,0)

def test_cuda_parity_when_available(profile):
    torch=pytest.importorskip('torch')
    if not torch.cuda.is_available():pytest.skip('CUDA unavailable; no GPU verification claimed')
    from atlas_drive_runtime.torch_backend import TorchEnvelope
    actual=TorchEnvelope(profile,device='cuda').evaluate([25.,-25.],[10.,-10.],52.5)
    expected=evaluate(profile,[25.,-25.],[10.,-10.],52.5)
    np.testing.assert_allclose(actual['torque_Nm'].cpu(),expected['torque_Nm'],rtol=1e-5)
