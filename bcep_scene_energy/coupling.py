"""One bounded coordinate sweep, hard acceptance, explicit rejected-proposal ST."""
import numpy as np
import torch
from torch import nn
from ebjd.unified import LearnableJointCouplingHead,log_sinkhorn,gather_candidates,permutation_matrix
from bcep_v2.joint import assignment
from bcep_v3.evidence import validate_inputs,CandidateSetEvidence
from .energy import SceneEnergy,ConditionalReplacementScorer
from .matching import solve
from .verification import EnergyVerifier


class CoordinatedAssignment(nn.Module):
    def __init__(self,rejection_gradient='proposal_st'):
        super().__init__()
        if rejection_gradient not in ('proposal_st','accepted_only'):raise ValueError('unknown ST policy')
        self.rejection_gradient=rejection_gradient;self.scorer=ConditionalReplacementScorer()

    def forward(self,state,pi,initial_st,initial_soft,*,straight_through,order=None):
        n,k=pi.shape
        order=list(range(n)) if order is None else list(order)
        if sorted(order)!=list(range(n)):raise ValueError('visit each valid person exactly once')
        hard_pi=pi.detach().cpu().numpy().copy();verifier=EnergyVerifier(state)
        st=[initial_st[i] for i in range(n)];soft=[initial_soft[i] for i in range(n)]
        scores=[initial_st[i]*0 for i in range(n)];trace=[]
        if n==1:return pi,initial_st,initial_soft,torch.stack(scores),trace,scores
        for i in order:
            s=self.scorer(state,torch.stack(st),i).float();scores[i]=s
            score_cpu=s.detach().cpu().numpy();current_i=hard_pi[i].copy()
            proposal,certificate=solve(score_cpu)
            proposed=hard_pi.copy();proposed[i]=proposal
            check=verifier.compare(hard_pi,proposed)
            changed=bool(np.any(proposal!=hard_pi[i]))
            accept=bool(changed and certificate['unique'] and check['reliable_decrease'])
            reason=('accepted' if accept else 'native_not_exactly_optimal' if not certificate['native_solution_exactly_optimal']
                    else 'strict_tie' if not certificate['unique'] else 'same_incumbent' if not changed
                    else 'nondecrease' if check['energy_delta_fp64']>=0 else 'roundoff_uncertain')
            if accept:hard_pi=proposed
            hard=torch.nn.functional.one_hot(torch.as_tensor(hard_pi[i],device=s.device),k).float()
            if straight_through:
                relaxed=log_sinkhorn(s,20)
                if accept or self.rejection_gradient=='proposal_st':
                    # Hard rejection still keeps exactly the incumbent. The adjoint
                    # uses the evaluated proposal scores to prevent rejection starvation.
                    st[i]=hard+(relaxed-relaxed.detach());soft[i]=relaxed
            else:st[i]=hard;soft[i]=hard
            predicted_delta=-float(score_cpu[np.arange(k),proposal].astype(np.float64).sum())
            trace.append(dict(person=i,accepted=accept,reason=reason,changed_proposal_slots=int(np.count_nonzero(proposal!=current_i)),
                incumbent_candidates=' '.join(map(str,current_i)),proposed_candidates=' '.join(map(str,proposal)),
                final_candidates=' '.join(map(str,hard_pi[i])),predicted_delta_from_FP32_scores=predicted_delta,
                FP64_delta_minus_score_prediction=check['energy_delta_fp64']-predicted_delta,
                **certificate,**check,energy_after_fp64=check['energy_proposed_fp64'] if accept else check['energy_before_fp64'],
                main_hungarian_calls=1))
        result=torch.as_tensor(hard_pi,device=pi.device)
        return result,torch.stack(st),torch.stack(soft),torch.stack(scores),trace,scores


class SceneEnergyCoupling(LearnableJointCouplingHead):
    def enable_energy(self,*,csee=False,rejection_gradient='proposal_st'):
        self.energy=SceneEnergy();self.coordinator=CoordinatedAssignment(rejection_gradient)
        self.csee=CandidateSetEvidence() if csee else None
        self.last_edge_count=0;self.last_diagnostics=[]

    def assign(self,context,proposals,goal_scale,*,straight_through,return_details=False,incumbent=None,visit_orders=None):
        validate_inputs(context,proposals)
        if context.valid.shape[1]>57:raise ValueError('validated capacity is at most 57 people')
        goals=proposals.detach();safe=torch.where(context.valid[:,None,:,None],goals,0.)
        s0,u=self.forward_features(context,safe,goal_scale)
        pi0,initial,soft0=assignment(s0,context.valid,straight_through)
        if incumbent is not None:
            if incumbent.shape!=pi0.shape or not torch.equal(incumbent.sort(1).values,pi0.sort(1).values):raise ValueError('invalid incumbent')
            for b in range(len(context.valid)):
                if int(context.valid[b].sum())==1 and not torch.equal(incumbent[b],pi0[b]):raise ValueError('single-person identity required')
            pi0=incumbent
            hard=permutation_matrix(pi0,20)
            initial=hard+(soft0-soft0.detach()) if straight_through else hard
        if self.csee is not None:u=self.csee(u,context,safe,goal_scale)
        states,positions=self.energy.prepare(u,context,safe,goal_scale)
        pis=[];softs=[];score_list=[];details=[]
        for b,(state,active) in enumerate(zip(states,positions)):
            order=None if visit_orders is None else visit_orders[b]
            p,st,soft,score,trace,score_rows=self.coordinator(state,pi0[b,:,active].T,initial[b,active],soft0[b,active],
                straight_through=straight_through,order=order)
            # index_copy is deterministic and preserves the differentiable valid rows.
            pi=pi0[b].T.index_copy(0,active,p)
            outsoft=soft0[b].index_copy(0,active,soft)
            outscore=s0[b].index_copy(0,active,score)
            pis.append(pi.T);softs.append(outsoft);score_list.append(outscore)
            details.append(dict(scene=b,trace=trace,valid_people=len(active),edges=len(state.edges),
                                initial_st=initial[b,active],final_st=st,state=state,proposal_scores=score_rows))
        pi=torch.stack(pis);soft=torch.stack(softs);scores=torch.stack(score_list)
        hard=gather_candidates(goals,pi)
        if straight_through:
            surrogate=torch.einsum('bnwa,bnad->bwnd',soft,safe.transpose(1,2))
            hard=hard+(surrogate-surrogate.detach())
        self.last_edge_count=sum(2*len(s.edges) for s in states)
        # Retain detached scalars only; do not pin a completed autograd graph.
        self.last_diagnostics=[dict(scene=d['scene'],valid_people=d['valid_people'],edges=d['edges'],trace=d['trace']) for d in details]
        result=(hard,pi,scores)
        if return_details:return (*result,dict(initial_pi=pi0,initial_scores=s0,final_soft=soft,scenes=details))
        return result
