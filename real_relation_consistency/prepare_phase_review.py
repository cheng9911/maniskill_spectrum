"""Make offline video review clips and a browser annotation tool; no labels invented."""
from pathlib import Path
import json,subprocess,concurrent.futures
BASE=Path(__file__).resolve().parent/'real_phase_identification';FF='/home/rocos/miniconda3/envs/lerobot/lib/python3.10/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux64-v4.2.2'
rows=json.loads((BASE/'video_review_manifest.json').read_text());(BASE/'clips').mkdir(exist_ok=True)
def make(r):
 if r['source_duration_mismatch']:return dict(dataset=r['dataset'],episode=r['episode'],status='excluded_duration_mismatch')
 out=BASE/r['clip']
 subprocess.run([FF,'-hide_banner','-loglevel','error','-c:v','libdav1d','-ss',str(r['video_start_s']),'-i',r['video'],'-t',str(r['clip_duration_s']),'-an','-vf','scale=480:-2,fps=15','-c:v','libx264','-threads','2','-preset','fast','-crf','22','-movflags','+faststart','-y',str(out)],check=True)
 return dict(dataset=r['dataset'],episode=r['episode'],status='clip_ready',path=r['clip'],bytes=out.stat().st_size)
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:status=list(pool.map(make,rows))
(BASE/'clip_export_status.json').write_text(json.dumps(status,indent=2)+'\n')
html=r'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>真机阶段视频复核</title>
<style>body{font:16px system-ui;margin:24px auto;max-width:1050px;color:#24313b}video{width:640px;max-width:100%;background:#111}table{border-collapse:collapse;width:100%}th,td{padding:8px;border-bottom:1px solid #ddd;text-align:left}input[type=number]{width:75px}button,select,input{padding:6px;margin:3px}small{color:#596875}.note{padding:12px;background:#fff1d7}</style>
<h1>真机阶段视频复核</h1><p>先独立观察视频，再标时间区间。自动候选默认隐藏。看不见轴尖接触时选“不可观察”，不要把轴向运动当成接触。剪辑降至 15 fps，原数据为 30 Hz；视频元数据时长一致不保证精确同步。</p>
<p class="note">此页面不操作机器人。尚未填写的项目不是人工真值。填写后导出 JSON，由评价脚本与自动候选比较。</p>
<label>标注者 <input id="rater" placeholder="输入标注者ID"></label><select id="episode"></select><button id="save">保存本条</button><button id="export">导出全部标注</button>
<div><video id="v" controls></video></div><p id="clock"></p><p>以剪辑秒数记录时间范围。明确的一帧可以设置相同上下界；遮挡时应扩大区间或标为不可观察。</p>
<table><thead><tr><th>事件</th><th>状态</th><th>下界(s)</th><th>上界(s)</th><th>取当前时间</th></tr></thead><tbody id="events"></tbody></table>
<p><label>备注 <input id="notes" style="width:70%"></label></p><button id="show">显示自动候选（之后该条标注标记为已看预测）</button><pre id="pred"></pre>
<script>
const manifest=__MANIFEST__, labels={grasp:'夹持闭合',lift_clearance:'抬离拾取位置',assembly_region:'进入装配接近区域',axial_advance_candidate:'可见轴向推进开始',contact:'首次可见接触',terminal_settle_candidate:'末段稳定',release:'释放'};
const v=document.getElementById('v'),pick=document.getElementById('episode');let store=JSON.parse(localStorage.getItem('phase_review_v1')||'{}'),shown=false;
function current(){return manifest[Number(pick.value)||0]}function key(r){return r.dataset+'/'+r.episode}function put(){const r=current();const events={};for(const name of Object.keys(labels)){let row=document.getElementById('row_'+name);events[name]={status:row.querySelector('select').value,lower_clip_s:row.querySelector('.lo').value===''?null:Number(row.querySelector('.lo').value),upper_clip_s:row.querySelector('.hi').value===''?null:Number(row.querySelector('.hi').value)}}store[key(r)]={dataset:r.dataset,episode:r.episode,rater:document.getElementById('rater').value,clip_start_frame:r.clip_start_frame,source_fps:r.fps,review_fps:15,predictions_shown:shown,notes:document.getElementById('notes').value,events};localStorage.setItem('phase_review_v1',JSON.stringify(store))}
function load(){const r=current(),old=store[key(r)];shown=old?.predictions_shown||false;v.src=r.clip;document.getElementById('pred').textContent='';document.getElementById('notes').value=old?.notes||'';document.getElementById('events').innerHTML='';for(const [name,label] of Object.entries(labels)){const e=old?.events?.[name]||{},row=document.createElement('tr');row.id='row_'+name;row.innerHTML=`<td>${label}</td><td><select><option value="unlabeled">未标注</option><option value="observed">可观察</option><option value="unobservable">不可观察</option></select></td><td><input class="lo" type="number" min="0" step="0.067"></td><td><input class="hi" type="number" min="0" step="0.067"></td><td><button class="low">下界</button><button class="high">上界</button></td>`;row.querySelector('select').value=e.status||'unlabeled';row.querySelector('.lo').value=e.lower_clip_s??'';row.querySelector('.hi').value=e.upper_clip_s??'';row.querySelector('.low').onclick=()=>row.querySelector('.lo').value=v.currentTime.toFixed(3);row.querySelector('.high').onclick=()=>row.querySelector('.hi').value=v.currentTime.toFixed(3);document.getElementById('events').append(row)}}
manifest.forEach((r,i)=>pick.add(new Option(r.dataset+' / '+r.episode,i)));pick.onchange=load;document.getElementById('save').onclick=put;document.getElementById('show').onclick=()=>{shown=true;const r=current();document.getElementById('pred').textContent=JSON.stringify(Object.fromEntries(Object.entries(r.events).map(([k,f])=>[k,f===null?null:(f-r.clip_start_frame)/r.fps])),null,2)};
document.getElementById('export').onclick=()=>{put();const blob=new Blob([JSON.stringify({schema:1,annotations:Object.values(store)},null,2)],{type:'application/json'}),a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='manual_phase_annotations.json';a.click();URL.revokeObjectURL(a.href)};
v.ontimeupdate=()=>document.getElementById('clock').textContent='剪辑 '+v.currentTime.toFixed(3)+' s；原数据约帧 '+Math.round(current().clip_start_frame+v.currentTime*current().fps);load();
</script></html>'''
(BASE/'review.html').write_text(html.replace('__MANIFEST__',json.dumps(rows)))
print(json.dumps(status,indent=2))
