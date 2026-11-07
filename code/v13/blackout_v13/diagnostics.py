"""Public-observation proxies, kept outside the policy and training objective."""
import numpy as np


class Diagnostics:
    def __init__(self):
        self.carry_start=np.full(5,np.nan);self.prediction=np.full(5,np.nan)
        self.counts=dict(pickups=0,return_proxies=0,carrying_loss_proxies=0,task_switches=0,
                         moving_unit_steps=0,stalled_unit_steps=0,manager_events=0,teacher_events=0,
                         teacher_supported_events=0,return_seconds_sum=0.,eta_absolute_error_sum=0.,eta_samples=0)

    def observe(self,old,new,sem,clear,event,teacher_supported=None):
        old=old.detach().cpu().numpy();new=new.detach().cpu().numpy()
        clear=clear.detach().cpu().numpy();stores=sem[2].detach().cpu().numpy()
        now=new[:,6];previous=old[:,7];holding=new[:,7]
        live=(old[:,11]>0)&~clear
        pickups=live&(previous==0)&(holding>0)
        self.carry_start[pickups]=now[pickups];self.prediction[pickups]=new[pickups,20]
        dropped=live&(previous>0)&(holding==0)
        cells=np.clip((new[:,9:11]*24).astype(int),0,23)
        at_store=stores[cells[:,1]*24+cells[:,0]]>0
        returns=dropped&at_store
        measured=returns&np.isfinite(self.carry_start)
        duration=now[measured]-self.carry_start[measured]
        self.counts['return_seconds_sum']+=float(duration.sum())
        self.counts['eta_absolute_error_sum']+=float(np.abs(duration-self.prediction[measured]).sum())
        self.counts['eta_samples']+=int(measured.sum())
        self.carry_start[dropped|clear]=np.nan
        attempted=live&(np.linalg.norm(old[:,13:15],axis=1)>.5)
        stationary=np.linalg.norm(new[:,9:11]-old[:,9:11],axis=1)<1e-5
        self.counts['pickups']+=int(pickups.sum())
        self.counts['return_proxies']+=int(returns.sum())
        self.counts['carrying_loss_proxies']+=int((dropped&~at_store).sum()+((previous>0)&clear&(old[:,11]>0)).sum())
        self.counts['task_switches']+=int((live&(new[:,12]!=old[:,12])).sum())
        self.counts['moving_unit_steps']+=int(attempted.sum())
        self.counts['stalled_unit_steps']+=int((attempted&stationary).sum())
        self.counts['manager_events']+=int(event)
        if event and teacher_supported is not None:
            self.counts['teacher_events']+=1;self.counts['teacher_supported_events']+=int(teacher_supported)

    def report(self):
        c=self.counts
        return dict(**c,scope='observation proxies, not engine-confirmed delivery events',
                    return_seconds=c['return_seconds_sum']/max(1,c['eta_samples']),
                    eta_mae_seconds=c['eta_absolute_error_sum']/max(1,c['eta_samples']),
                    stalled_fraction=c['stalled_unit_steps']/max(1,c['moving_unit_steps']))
