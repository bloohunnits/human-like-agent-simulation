import fs from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL, fileURLToPath } from 'node:url';
import { Presentation, PresentationFile } from '@oai/artifact-tool';

const ROOT=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'../..');
const BUILD=path.join(ROOT,'.proposal-build');
const OUT=path.join(ROOT,'output/presentations');
const SKILL=process.env.PRESENTATIONS_SKILL_DIR||'/Users/marshall/.codex/plugins/cache/openai-primary-runtime/presentations/26.909.12148/skills/presentations';
const PY=process.env.RUNTIME_PYTHON||'/Users/marshall/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3';
process.env.RUNTIME_NODE ||= process.execPath;
process.env.RUNTIME_NODE_MODULES ||='/Users/marshall/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules';
process.env.RUNTIME_BIN_DIR ||='/Users/marshall/.cache/codex-runtimes/codex-primary-runtime/dependencies/bin/override';
process.env.RUNTIME_PYTHON=PY;
const {resolvePresentationFont,applyPresentationChartFont,finalizePresentation}=await import(pathToFileURL(path.join(SKILL,'container_tools/artifact_tool_utils.mjs')).href);
const FONT=resolvePresentationFont({fontFamily:'Arial'});
const C={bg:'#F7F5EF',ink:'#142F36',muted:'#53656A',teal:'#087F82',amber:'#B76C26',light:'#E4E9E5',white:'#FFFFFF'};
const p=Presentation.create({slideSize:{width:1280,height:720}});
const notes=[];
const tableOwners=[];
const chartOwners=[];
const src={
 proposal:ROOT+'/proposal/proposal.pdf',
 kernel:ROOT+'/code/hou_memory/hou_memory.py',
 graph:ROOT+'/code/hou_memory/associative_memory.py',
 agent:ROOT+'/code/hou_memory/memory_agent.py',
 web:ROOT+'/code/hou_memory/webapp.py',
 prompt:ROOT+'/code/terralingua/core/agents/prompt_templates.py',
 terraCode:ROOT+'/code/terralingua/core/agents/llm_agent.py',
 hou:'Hou, Tamoto & Miyashita. CHI EA 2024. https://doi.org/10.1145/3613905.3650839',
 honda:'Honda et al. HAI 2025 proceedings, published 2026. https://doi.org/10.1145/3765766.3765803',
 terra:'Paolo et al. TerraLingua (2026). https://arxiv.org/abs/2603.16910',
 apa:'APA Dictionary of Psychology, Associative Memory. https://dictionary.apa.org/associative-memory',
 emotion:'Huang et al. EmotionBench. https://arxiv.org/abs/2308.03656',
 shachi:'Kuroki et al. Shachi. https://arxiv.org/abs/2509.21862'
};
function text(s,value,x,y,w,h,size=28,{color=C.ink,bold=false,align='left'}={}){
 const a=s.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
 a.text=value;a.text.style={typeface:FONT,fontSize:size,color,bold,alignment:align,verticalAlignment:'top',autoFit:'none',wrap:'square',insets:{left:0,right:0,top:0,bottom:0},lineSpacing:1.12};return a;
}
function slide(title,{dark=false,backup=false}={}){
 const s=p.slides.add();s.background.fill=dark?C.ink:C.bg;
 if(title)text(s,title,70,54,1135,104,43,{bold:true,color:dark?C.white:C.ink});
 text(s,`${backup?'Backup  ':''}${p.slides.items.length}`,1100,671,110,26,18,{align:'right',color:dark?'#BED2D4':C.muted});return s;
}
function note(s,title,body,sources=[]){sources=sources.map(v=>v.startsWith(ROOT+'/')?'https://github.com/bloohunnits/human-like-agent-simulation/blob/main/'+v.slice(ROOT.length+1):v);s.speakerNotes.textFrame.setText(`${body}\n\nSources\n${sources.join('\n')}`);notes.push({number:p.slides.items.length,title,body,sources});}
function foot(s,t,y=626){text(s,t,70,y,1090,38,19,{color:C.muted});}
function table(s,values,{x=70,y=184,w=1140,h=350,widths,size=25}={}){
 tableOwners.push(p.slides.items.length);
 const t=s.tables.add({rows:values.length,columns:values[0].length,left:x,top:y,width:w,height:h,values,columnWidths:widths});
 t.styleOptions={headerRow:true,bandedRows:false};t.borders.assign({fill:C.light,width:.8,style:'solid'});
 for(let r=0;r<values.length;r++)for(let c=0;c<values[r].length;c++){const cell=t.getCell(r,c);cell.fill=r===0?C.ink:C.bg;cell.text.style={typeface:FONT,fontSize:size,color:r===0?C.white:C.ink,bold:r===0,verticalAlignment:'middle',insets:{left:16,right:12,top:12,bottom:12},autoFit:'none'};}
 return t;
}
function chart(s,type,options){chartOwners.push(p.slides.items.length);const ch=s.charts.add(type,options);applyPresentationChartFont(ch,{fontFamily:FONT});return ch;}
function pair(s,heading,body,x,y,w=520){text(s,heading,x,y,w,48,31,{bold:true,color:C.teal});text(s,body,x,y+62,w,190,28);}

const design=ROOT+'/docs/INDEPENDENT_GRAPH_MEMORY_DESIGN.md';
const example=JSON.parse(await fs.readFile(path.join(ROOT,'output/analysis/proposal-examples-v10.json'),'utf8'));
const pinwheel=JSON.parse(await fs.readFile(path.join(ROOT,'output/analysis/pinwheel-walkthrough.json'),'utf8'));
const F=x=>-Math.expm1(-x)/(-Math.expm1(-1));
const fmt=x=>x.toFixed(3);
const fan='Anderson & Reder (1999), The Fan Effect: New Results and New Theories. https://www.andrew.cmu.edu/user/reder/publications/99_jra_lmr_2.pdf';
function lineChart(s,categories,series,{x=440,y=173,w=768,h=421,max=.7,xTitle='Ticks since encoding'}={}){
 chart(s,'line',{position:{left:x,top:y,width:w,height:h},categories:categories.map(String),series:series.map((a,i)=>({name:a.name,values:a.values.map(v=>Number(v.toFixed(12))),line:{fill:a.color||[C.ink,C.teal,C.amber,'#849397'][i],width:3},marker:{symbol:'none'}})),hasLegend:true,legend:{position:'bottom',textStyle:{typeface:FONT,fontSize:21}},xAxis:{title:{text:xTitle,textStyle:{typeface:FONT,fontSize:22}},textStyle:{typeface:FONT,fontSize:19},majorGridlines:null},yAxis:{min:0,max,majorUnit:.1,title:{text:'Retrieval score',textStyle:{typeface:FONT,fontSize:22}},numberFormatCode:'0.0',textStyle:{typeface:FONT,fontSize:19},majorGridlines:{fill:C.light,width:1}},chartFill:C.bg,plotAreaFill:C.bg});
}
// 1
{
const s=slide('',{dark:true});
text(s,'Relational memory\nin multi-agent simulations',70,139,1110,214,66,{bold:true,color:C.white});
text(s,'Modeling human-like memory\nfor more believable simulations',74,389,1080,106,35,{color:'#D0E3E2'});
text(s,'Andre Atkins, Ben Sadorra, and Ryan Shechtman\nCMSC473/673',74,590,990,68,25,{color:'#D0E3E2'});
note(s,'Relational memory in multi-agent simulations',`Our goal is a simulation whose agents remember and forget in more believable ways. We propose a separately fading associative pathway, so a context can bring back an old experience even when ordinary retrieval is weak. The connection also fades and can strengthen through use. We will test the mechanism first and assess simulation believability separately.`,[src.proposal,src.hou,design]);
}
// 2
{
const s=slide('The problem: human-like remembering');
text(s,'A familiar place can bring back an old experience.',70,174,1120,55,36,{bold:true});
text(s,'The pattern we want to model',70,282,510,44,30,{bold:true,color:C.teal});
text(s,'An experience can be hard to recall.\nA specific context can bring it back.\nOther experiences continue to fade.',70,344,515,167,28);
text(s,'The gap in a simple retrieval rule',665,282,530,44,30,{bold:true,color:C.teal});
text(s,'Text similarity finds matching content.\nRecency favors recent experiences.\nPersonal associations need their own history.',665,344,530,167,28);
text(s,'Our goal is believable remembering and forgetting in a social simulation.',70,565,1120,82,34,{bold:true,color:C.teal});
note(s,'The problem: human-like remembering',`The supplied proposal makes believability the purpose of the memory model. A place, person or event can act as an associative cue even when an old episode has not been recalled recently. We want selective reminders and plausible lapses rather than maximum recall of everything. The exact equations and learning rules here are project hypotheses. This project does not establish a general improvement in agent intelligence, efficiency or task performance. Human ratings will assess simulation believability; mathematical retrieval examples only establish what the proposed mechanism can do.`,[src.proposal,src.apa,design]);
}
// 3
{
const s=slide('Example: remembering where Mia helped');
text(s,'Earlier',70,179,190,46,31,{bold:true,color:C.teal});
text(s,'Mia gives the agent food beside a blue pinwheel.\nShe says, “I’m usually here.”',295,179,910,99,31);
text(s,'Much later',70,309,205,46,31,{bold:true,color:C.teal});
text(s,'The hungry agent notices the same pinwheel.\nMia and food are outside its current view.',295,309,910,99,31);
text(s,'The reminder',70,439,220,46,31,{bold:true,color:C.teal});
text(s,'The pinwheel brings back that encounter,\ngiving the agent a reason to look for Mia there.',295,439,910,99,31);
text(s,'Expected behavior: approach the location and seek Mia’s help.',70,586,1130,62,32,{bold:true,color:C.teal});
note(s,'Example: remembering where Mia helped',`Mia gives the agent food beside a uniquely named blue pinwheel and says only “I'm usually here.” She never instructs the agent to return for food. Much later, the hungry agent sees the pinwheel without seeing Mia or food. Recovering the earlier episode supplies grounds for a reasonable inference: Mia helped before and might be found here again. The expected action is to approach and look for her; it does not assert that food or Mia is currently there. A different personal history could support another action. The relation connects the observed artifact to the episode; it is not a trained reward value or a universal pinwheel-means-food rule.

The example is a constructed one-episode fixture, scored with the actual MiniLM encoder, episode writer and production retrieval engine. Food-transfer feedback is supplied by the fixture, not verified by a world rollout. The approach behavior has not been tested with an LLM. The stored text includes blue pinwheel and food, so semantic similarity is positive. With hunger explicitly appended to the query, raw cosine is .3702016175. At 100 ticks, direct final score is .0106903 and associative final score is .2164730. The point is loss of access under similarity plus ordinary decay, not that vectors can never recognize the content.

The current live query normally includes observations, messages and optional information; energy reaches the action model separately. This fixture additionally includes “I am hungry. Current goal: find food.” as a stronger comparison. That query change is a fixture only, not a change to the simulation. Seeing a cue can trigger graph access today; hunger alone does not activate an unseen pinwheel. The later chart gives the exact accessibility window.`,[src.agent,src.graph,ROOT+'/output/analysis/pinwheel-walkthrough.json',ROOT+'/scripts/pinwheel_walkthrough.py']);
}
// 4
{
const s=slide('Thesis and testable hypotheses');
text(s,'Associative memory should give agents more appropriate\ncontext for their decisions, making their behavior more\nbelievable and consistent with their own experiences.',70,162,1140,139,34,{bold:true});
text(s,'H1  Associative recovery',70,344,540,47,31,{bold:true,color:C.teal});
text(s,'Associations will help agents retrieve\npast experiences relevant to their\npresent situation, including experiences\nthat semantic similarity and time decay\nalone would miss.',70,408,540,187,27);
text(s,'H2  Lasting accessibility',665,344,535,47,31,{bold:true,color:C.teal});
text(s,'Recalling experiences through their\nassociations will help preserve relevant\ncontext across longer gaps.',665,408,535,148,28);
note(s,'Thesis and testable hypotheses',`The thesis is about appropriate context and believable behavior in a simulation. H1 asks whether associative retrieval recovers relevant episodes that the semantic-plus-decay baseline misses under the same prompt budget. H2 asks whether using those connections preserves relevant access across later gaps. Both require useful selectivity, not a higher score alone. Their directional score effects are designed into the rule. The empirical question is whether those changes recover useful context across held-out histories without excessive false reminders or displacement.\n\nTime is a variable: sweep the delay between experience or last recall and the probe. At each delay, compare copies of the same state so one model does not get younger or stronger memories. For H2, give both copies the same ordinary recall events and vary only edge learning. A separate closed-loop simulation then lets histories diverge naturally to measure the whole system. Human ratings are necessary for the believability claim.\n\nOperationally, test improvements across a prespecified collection of histories rather than selecting only successful examples. Label relevant episodes before observing retrieval. Count distracting reminders and displacement alongside relevant recall. Choose an acceptable false-reminder limit on development cases and freeze evaluation criteria before held-out testing. Prompt inclusion is not demonstrated use by the model, and context-appropriate behavior is not identical to human-like memory.`,[src.proposal,src.graph,design]);
}
// 5
{
const s=slide('Models and their roles');
table(s,[['Role','Planned model','Rationale'],['Direct similarity','SBERT all-MiniLM-L6-v2','Sparser similarity scores in the cited setup.\nSmall enough to run locally.'],['Actions and messages','GPT-5 Nano','API generation for repeated\nmatched simulation runs.'],['Lower API-cost option','A locally available Qwen model','Local generation for larger collections\nwhen the hardware permits.']],{y:183,h:336,widths:[285,345,510],size:25});
text(s,'SBERT scores the cue against memory text.\nThe generation model receives memories before acting.',70,552,1120,90,31,{bold:true,color:C.teal});
note(s,'Models and their roles',`The primary reason for MiniLM in the supplied proposal is its relatively sparse similarity scores in Honda's setting. This concerns the distribution of similarity scores, not sparse vector coordinates or a guarantee on our own data. Local execution is another benefit. Qwen is a practical lower API-cost option for collecting results, not a mandatory cross-model hypothesis. Hold the generation model constant within each comparison. The exact locally available Qwen version is a deployment choice; the proposal and current router name different versions. We train no model weights, but still need labeled evaluation cases.`,[src.proposal,src.honda,ROOT+'/code/hou_memory/embedder.py',design]);
}
// 6
{
const s=slide('The memory’s own forgetting curve');
text(s,'Direct relevance rᵢ',70,178,355,42,29,{bold:true,color:C.teal});
text(s,'Cue-to-memory\nembedding similarity',70,225,345,86,27);
text(s,'Elapsed time',70,337,345,42,29,{bold:true,color:C.teal});
text(s,'Time since full recall',70,384,345,80,27);
text(s,'Memory strength gᵢ',70,481,355,42,29,{bold:true,color:C.teal});
text(s,'Each memory has its own g.\nA higher g slows forgetting.',70,528,350,86,27);
const ticks=Array.from({length:11},(_,i)=>i*15);
lineChart(s,ticks,[1,2,4].map((g,i)=>({name:`g = ${g}`,values:ticks.map(t=>F(.6*Math.exp(-t/(25*g)))),color:[C.ink,C.teal,C.amber][i]})),{max:.8,xTitle:'Ticks since full recall'});
foot(s,'Calculated Hou curves at relevance .60. These are model scores, not measured human recall rates.',637);
note(s,'The memory’s own forgetting curve',`Hou combines direct relevance with time since recall and memory-specific strength. Full recall adds tanh(elapsed/(2 tau_m)) to g and resets the recall timestamp. Memories can start with equal g and diverge through different recall histories. The graph extension retains this direct-memory pathway. The illustrated curves use r=.60, tau_m=25 ticks, no intervening recalls and g values 1, 2 and 4. The score mapping is monotonic and our implementation selects deterministically; the chart is not a human probability estimate.`,[src.hou,src.kernel,design]);
}
// 7
{
const s=slide('A separate pathway for associations');
text(s,'Direct-memory contribution',70,178,540,44,29,{bold:true,color:C.teal});
text(s,'Mᵢ = rᵢ exp(−tᵢ / (τₘ gᵢ))',70,240,540,62,33,{bold:true});
text(s,'rᵢ  cue-to-memory cosine similarity\ngᵢ  this memory’s strength\nτₘ  tick-to-memory time scale',70,324,530,128,27);
text(s,'One uncrowded connection',665,178,550,44,29,{bold:true,color:C.teal});
text(s,'Gᵢ = β c w exp(−Δ / h)',665,240,540,62,33,{bold:true});
text(s,'c  current cue activation\nw  connection weight\nh  connection retention time',665,324,530,128,27);
text(s,'xᵢ = max(Mᵢ, Gᵢ)',70,504,530,58,37,{bold:true,color:C.teal});
text(s,'pᵢ = (1 − exp(−xᵢ)) / (1 − exp(−1))',665,510,545,58,29,{bold:true});
foot(s,'Each route ages separately. Max takes the stronger contribution. β limits the graph contribution.');
note(s,'A separate pathway for associations',`This slide shows the one-edge, uncrowded case. On the slide t_i is elapsed ticks since the memory's full recall; the design brief uses t minus the last-recall timestamp. Delta is elapsed ticks since the edge's last qualifying reinforcement. tau_m converts ticks to Hou model time. g is memory-specific; w and h belong to the edge. c is current source activation, zero for an absent source, and beta is a global graph budget. All contributions are bounded. First decay each route separately, then take max and apply the monotonic Hou score mapping. Do not apply the target decay a second time. A weak positive connection does not count as evidence against retrieval. Max ignores it when direct access is stronger. This model has no inhibitory links; several weak cues also cannot add together. General routes and crowding are in the backup equations. This is a proposed Hou-based extension, not a published human-memory formula.\n\nThe visible equation is the uncrowded single-edge case. The current general rule also multiplies by f(n)=1 for n<=3 and 1/(1+ln(n/3)) otherwise. Each hub counts all retained episode edges, including faded ones. Once G exceeds M, changing direct relevance while M stays below G has no effect on this score. Max can therefore erase semantic distinctions among linked episodes. Its positive-support interpretation is internally consistent, but selectivity must be evaluated.`,[design,src.kernel]);
}
// Additional worked-example chart, after the two mathematical explanation slides.
{
const s=slide('When the pinwheel can bring it back');
text(s,'Ticks 0–32',70,177,300,43,30,{bold:true,color:C.teal});
text(s,'Either method can\nretrieve the encounter.',70,229,300,86,27);
text(s,'Ticks 33–139',70,342,300,43,30,{bold:true,color:C.teal});
text(s,'Only the association\nkeeps it above .15.',70,394,300,86,27);
text(s,'From tick 140',70,506,300,43,30,{bold:true,color:C.teal});
text(s,'Both fall below\nthe cutoff.',70,558,300,75,27);
const rows=pinwheel.curve;
const start=Math.round(pinwheel.direct_threshold_tick/180*100000);
const end=Math.round(pinwheel.association_threshold_tick/180*100000);
chart(s,'scatter',{
 position:{left:388,top:174,width:824,height:438},
 scatterOptions:{style:'line'},
 series:[
  {name:'Similarity + decay',xValues:rows.map(a=>a.tick),values:rows.map(a=>Number(a.direct_score.toFixed(12))),line:{fill:C.ink,width:3},marker:{symbol:'none'}},
  {name:'With association',xValues:rows.map(a=>a.tick),values:rows.map(a=>Number(a.with_association_score.toFixed(12))),line:{fill:C.teal,width:3},marker:{symbol:'none'}},
  {name:'Cutoff .15',xValues:rows.map(a=>a.tick),values:rows.map(()=>.15),line:{fill:C.amber,width:2},marker:{symbol:'none'}}
 ],
 hasLegend:true,legend:{position:'bottom',textStyle:{typeface:FONT,fontSize:20}},
 xAxis:{min:0,max:180,majorUnit:30,title:{text:'Ticks since the encounter',textStyle:{typeface:FONT,fontSize:22}},textStyle:{typeface:FONT,fontSize:19},majorGridlines:null},
 yAxis:{min:0,max:.6,majorUnit:.15,title:{text:'Final retrieval score',textStyle:{typeface:FONT,fontSize:22}},numberFormatCode:'0.00',textStyle:{typeface:FONT,fontSize:19},majorGridlines:{fill:C.light,width:1}},
 chartFill:C.bg,
 plotAreaFill:{type:'gradient',gradientKind:'linear',angleDeg:0,stops:[
  {offset:0,color:C.bg},{offset:start-1,color:C.bg},{offset:start,color:'#E0ECE7'},
  {offset:end,color:'#E0ECE7'},{offset:end+1,color:C.bg},{offset:100000,color:C.bg}
 ]}
});
foot(s,'One fixed episode and visible cue. No intervening recall, new connections or competing memories.',647);
note(s,'When the pinwheel can bring it back',`All points are independent, read-only probes of the same unmodified memory. Both curves use the final retrieval score, not raw cosine or a measured probability. The hunger-aware query has r=.37020161747932434. M(t)=r exp(-t/25); G(t)=.5 × .8 × exp(-t/100). Both pass through F(x)=(1-exp(-x))/(1-exp(-1)); the associative condition uses F(max(M,G)). The node starts at g=1 and the edge at weight .8 and retention 100 ticks, with both clocks at zero. One active observed pinwheel route has no crowding penalty.

With p>.15 required, the equivalent pre-mapping cutoff is .0996193431. Ordinary access reaches the cutoff at 32.8172854 ticks and associative access at 139.0108193 ticks. Thus both qualify at integer ticks 0–32, only the association at 33–139, and neither from 140 onward. The native chart shades the continuous interval between the two threshold crossings. At tick 100 the direct score is .0106903 and graph-supported score is .2164730.

There is NO positive-time crossing of the direct and graph contributions in this example: G(0)=.4 already exceeds M(0)=.3702 and its retention time is longer. Solving equality gives a negative, out-of-domain time (-2.58056 ticks). Do not claim that the graph begins weaker and later overtakes direct similarity. The relevant transition is loss of prompt eligibility. Raw cosine .3702 stays constant for fixed texts; it must not be compared directly to the final .2165 score.

An unrehearsed direct route with even r=1 has final score only .02871 at tick 100 with g=1 and tau=25, so this fixture mainly illustrates the chosen forgetting timescales. A slower ordinary-decay baseline is needed to isolate association specificity in the experiment. No parameter or production query changes were made. Rehearsal, new hub members and competition would alter the displayed window and are separate experiments.`,[src.graph,src.kernel,ROOT+'/output/analysis/pinwheel-walkthrough.json',ROOT+'/scripts/pinwheel_walkthrough.py']);
}
// 8
{
const s=slide('Which memories enter the prompt?');
const rows=[
['1','Score the memories','Compare direct and associative contributions before changing any strengths.'],
['2','Fill the memory slots','Take up to 3 highest-scoring memories above .15 that fit within 600 tokens.'],
['3','Apply recall updates','Rehearse each included memory. Update its strongest graph route only if it beats direct access.']
];
rows.forEach((r,i)=>{const y=178+i*135;text(s,r[0],70,y,65,61,44,{bold:true,color:C.teal});text(s,r[1],155,y,388,100,30,{bold:true});text(s,r[2],598,y,600,112,28);});
text(s,'“Selected” means its text enters the agent’s prompt.',70,603,1120,53,32,{bold:true,color:C.teal});
note(s,'Which memories enter the prompt?',`Yes, selection includes top-K. The exact rule ranks by score, requires p strictly greater than .15, and returns at most three memories under a total 600 cl100k-base retrieved-text token budget. A too-long memory can be skipped so another eligible shorter memory fits. Fewer than three, including none, may enter the prompt. These are configurable starting values. The plan has a separate 60-token cap.\n\nEvery returned memory gets ordinary Hou rehearsal: its own g increases according to spacing and its last-full-recall timestamp resets. Its association learns only if G exceeds M. A direct-route tie gets no graph credit. Update the strongest winning route after selection and deduplicate shared edges once per event. A high-scoring candidate that fails the cutoff, count limit or token limit receives no rehearsal. Simply observing the pinwheel creates no renewal of every old pinwheel link.\n\nSelected, returned, surfaced and fully recalled refer to the same prompt inclusion event in this implementation. To avoid ambiguity we say enters the prompt. A route can receive credit even if that memory would also fit without graph support. Prompt inclusion does not prove the LLM used the memory in its action, so the audit records the exact prompt and response.\n\nThe implementation commits rehearsal for the retrieval event before invoking the language model. Retries reuse that committed selection and cannot double-rehearse. The update is therefore credit for assembling the memory input, not a reward for a correct or successful action. A generation failure does not automatically undo this retrieval event. Later graph-off probes must use matched copies: earlier graph-induced recalls may already have changed ordinary memory strength.`,[src.graph,src.agent]);
}
// 9
{
const s=slide('Connection rehearsal and later access');
text(s,'Both recall the same\nepisode once',70,184,345,89,30,{bold:true,color:C.teal});
text(s,'Only one also strengthens\nthe connection.',70,307,345,98,27);
text(s,'The score clears the\ncutoff longer.\nBoth links still fade.',70,466,345,122,27);
const rows=example.lasting_connection.rows;
lineChart(s,rows.map(v=>v.gap),[
{name:'No link update',values:rows.map(v=>v.no_link_update),color:C.ink},
{name:'Link strengthened once',values:rows.map(v=>v.one_link_update),color:C.teal},
{name:'Cutoff .15',values:rows.map(v=>v.cutoff),color:C.amber}
],{max:.5,xTitle:'Ticks after the recall'});
foot(s,'Fixed graph and cue. Separate probes, with no new episodes or intervening recalls.',637);
note(s,'Connection rehearsal and later access',`This chart compares use of a connection within the proposed architecture. Both copies recall the episode at tick 60, so their ordinary memory strength g=1.833655 and last-recall timestamp=60 are identical. Both continue to age over the horizontal axis. Only the teal copy updates its winning graph edge at that recall. The direct relevance is .20 and all other starting settings match.\n\nThe edge starts at w=.8 and h=100 ticks. Before recall its decayed transmission is .439049. Learning sets w=.551239, h=125 and its edge clock to 60. The no-learning copy retains the original edge history. The chart starts immediately after that recall and shows the full retrieval score with the same bell cue. The associative path dominates both curves for these inputs. At a gap of 90 ticks, the scores are .135 without link learning and .199 with it. The direct-only score is .044 in both. At longer gaps both fall below .15. The plotted probes do not commit any additional recall. This demonstrates the mechanism, not an empirically established duration of human memory.\n\nThis chart proves only that the configured link update increases future support with fixed topology, cue and ordinary memory history. It does not establish useful context. Adding episodes to the same hub changes its crowd factor, and competing episodes can exclude this memory from the prompt even above the cutoff. Do not interpret the curve as a live trajectory where every point generates and stores another episode.`,[src.graph,ROOT+'/output/analysis/proposal-examples-v10.json']);
}
// 10
{
const s=slide('Experiment conditions');
table(s,[['Condition','What it tests'],['Semantic similarity and ordinary decay','Baseline remembering and forgetting'],['Associative retrieval, link learning off','H1: relevant experiences recovered\nthrough surviving connections'],['Associative retrieval, link learning on','H2: useful context preserved after\na connection helps recall an experience']],{y:181,h:306,widths:[565,575],size:26});
text(s,'We will vary the time gap and the number of competing memories.',70,528,1130,83,32,{bold:true,color:C.teal});
foot(s,'Additional checks: goal-aware queries, slower ordinary forgetting, wrong cues and crowded places.');
note(s,'Experiment conditions',`The three presentation conditions correspond to launcher presets hou, independent and learned. All are implemented. The main proposal explains one associative architecture with learning off or on; earlier formula-placement comparisons belong in the design and iteration log.\n\nFor H1, take the same memory state at each delay and compare direct-only access with associative access while link learning is off. The time gap varies across tests. For H2, provide equal ordinary recall events to both copies while only one updates connections, then probe with the same cue after increasing gaps. Disable graph support in a read-only final probe to confirm equal direct scores. Frozen replay isolates mechanism effects. Closed-loop simulations subsequently measure the total effect as actions and histories diverge.\n\nChoose parameters on development histories and freeze them for held-out evaluation. Define relevant episodes from the scenario before scoring. Count irrelevant reminders and relevant episodes displaced from limited slots. Wrong cues, shuffled observed-entity links, hub crowding and paraphrases test specificity.\n\nThe primary study tests a memory policy. With memory_sources=false, a weighted cue-to-episode lookup table exactly reproduces the graph contribution. Multi-hop graph reasoning and a unique advantage of graph data structures are not tested. The graph-on comparison also changes the effective forgetting timescale, so include a slower-forgetting direct baseline and a goal-aware query as stronger checks before attributing every benefit to associations. Apply the same enriched query to both compared treatments. The simple weighted index is an equivalence control, not a different cognitive mechanism. Additional checks can stay out of the main three-condition table.`,[src.graph,ROOT+'/code/hou_memory/run_hou_experiment.py',design]);
}
// 11
{
const s=slide('TerraLingua and the memory audit UI');
s.images.add({blob:await fs.readFile(path.join(OUT,'terralingua-preview.gif')),contentType:'image/gif',fit:'contain',position:{left:70,top:206,width:560,height:350},alt:'TerraLingua environment preview'});
text(s,'The agent’s input',690,177,490,42,29,{bold:true,color:C.teal});
text(s,'Local observations and messages\nState, plan and recalled experiences',690,231,500,101,27);
text(s,'Planned audit trail',690,371,490,42,29,{bold:true,color:C.teal});
text(s,'Direct and associative contributions\nWinning route and edge age\nMemories included and left out\nChanges to memory and link strength',690,426,500,168,26);
foot(s,'Trace each retrieval and state update, then inspect the exact prompt and resulting action.');
note(s,'TerraLingua and the memory audit UI',`TerraLingua provides the social simulation. The looping GIF is an environment example, not evidence for the proposed treatment. PowerPoint plays it in Slide Show mode; the PDF shows one frame. We plan an interface that makes retrieval decisions observable, traceable and debuggable: M, G, x, p, winning route, edge w/h/age, target g and clock, cutoff, rank, prompt inclusion and state updates. Log the exact prompt and action so we can investigate unexpected behavior. The UI cannot prove that the LLM used a particular memory. The source cue uses local observations and messages rather than the entire world's events. Keep plans and any recent-history buffer consistent between conditions. k limits memory count, so also cap retrieved tokens.`,[src.terra,src.agent,src.web,design]);
}
// 12
{
const s=slide('Evaluation: memory patterns and believability');
table(s,[['Question','Evidence we will collect'],['H1: does context recover relevant old episodes?','Relevant recall, false reminders and displacement'],['H2: does rehearsal preserve useful access?','Useful and distracting reminders\nafter longer gaps'],['Do decisions fit the recalled context?','Prompt and action traces, plus blind ratings'],['Does the simulation feel more believable?','Blind ratings of remembering and behavior']],{y:182,h:341,widths:[610,530],size:24});
text(s,'Prompt inclusion, appropriate decisions and believability are separate outcomes.',70,559,1130,68,32,{bold:true,color:C.teal});
note(s,'Evaluation: memory patterns and believability',`A worked score crossing verifies the mechanism, not human-memory validity. Define labeled histories and meaningful cues before inspecting outcomes. Include wrong cues, no cue, crowded hubs, old irrelevant events and misleading temporal neighbors. Measure relevant and irrelevant retrieval together under the same k and token budget. For believability, blind raters to condition, randomize presentation order and use anchored judgments of continuity, lapses and associative reminders. Keep the generation model fixed within comparisons and report uncertainty and rater agreement. Inspect whether actions reflect the relevant remembered context as a secondary behavioral measure. Ordinary task accuracy is a diagnostic, not the thesis. The model retrieves stored text exactly; content distortion and clinical trauma mechanisms are outside its claims.\n\nUse three separate outcome levels: what enters the prompt, how the next action fits the agent’s available context, and whether independent raters find the remembering and behavior believable. A higher score proves none of these automatically. Raters should see relevant prior experience when judging context fit but remain blind to the memory condition. Similarity scores must not serve as the ground-truth relevance labels. The bounded plan can still carry old information, so audit plan content and match its rules across conditions.`,[design,src.proposal,src.shachi]);
}
// 13
{
const s=slide('Proposed implementation and experiments');
const rows=[['1','Reference scoring and traces','Separate the memory and edge clocks. Verify the worked examples.'],['2','Controlled comparisons','Vary delays and distractions. Test retrieval and connection learning.'],['3','Simulation and human review','Integrate the audit UI and evaluate believable remembering.']];
rows.forEach((r,i)=>{const y=182+i*128;text(s,r[0],70,y,65,61,44,{bold:true,color:C.teal});text(s,r[1],155,y,407,100,30,{bold:true});text(s,r[2],610,y,590,105,28);});
text(s,'Initial scope: independent associative access and connection learning.',70,596,1130,57,30,{bold:true,color:C.teal});
note(s,'Proposed implementation and experiments',`Build a versioned edge record and a pure scoring function before integrating the simulation. Check graph-off reduction, score bounds, absent-cue behavior, unused-edge forgetting, save/load, read-only previews and exactly-once updates. Then compare associative retrieval against the semantic-plus-decay baseline for H1, and connection learning on versus off with equal ordinary recall events for H2. Integrate stable place tags, source policies, memory granularity and token limits. Use the observed-entity route policy for the main experiments. Reserve additional mechanisms for later design work. The equations and examples describe the proposed system. Treat formula verification as a mechanism check and held-out selectivity as the experiment.`,[design,src.agent,src.graph]);
}
// 14
{
const s=slide('Full retrieval notation',{backup:true});
text(s,'q: current cue     mᵢ: stored episode     E: SBERT embedding',70,161,1130,44,27,{color:C.teal,bold:true});
text(s,'rᵢ = max(0, cos(E(q), E(mᵢ)))',70,226,1120,46,31,{bold:true});
text(s,'Mᵢ = rᵢ exp(−(t − tᵢ,last) / (τₘ gᵢ))',70,294,1120,46,31,{bold:true});
text(s,'Lₑ = wₑ exp(−(t − tₑ,last) / hₑ)',70,362,1120,46,31,{bold:true});
text(s,'Gᵢ = β max over routes [source cue × edge transmissions × crowd penalty]',70,430,1120,71,27,{bold:true});
text(s,'xᵢ = max(Mᵢ, Gᵢ)       pᵢ = (1 − exp(−xᵢ)) / (1 − exp(−1))',70,523,1130,53,29,{bold:true,color:C.teal});
foot(s,'The prompt receives at most k memories above the cutoff that fit within the token budget.');
note(s,'Full retrieval notation',`The exact general route formula is G_i=beta max_P[c_source product_(e in P)L_e f_P]. If no route exists, G=0. Apply beta once per allowed route and a crowd penalty once at a traversed hub. On a route through two physical edges, multiply both aged transmissions. Use r, c, w, beta and f in [0,1], positive time scales, and no future-dated records. The proposed score exactly reduces to Hou with G=0 for the same memory state. F is monotonic, so it changes score display and threshold scale but not rank. To show functional forgetting, tune a positive cutoff; use the same positive cutoff and prompt budget in every condition.`,[design,src.kernel]);
}
// 15
{
const s=slide('What changes after a retrieval',{backup:true});
table(s,[['State','Update','When'],['Memory strength g and clock','Increase g by the Hou spacing rule.\nReset its last-full-recall time.','The memory enters the prompt.'],['Connection strength w','Start from current transmission z.\nSet w = z + η(1 − z).','Its memory enters the prompt\nand this graph route beats direct access.'],['Connection duration h and clock','Raise h to a finite cap.\nRecord this reinforcement time.','Once per qualifying edge\nper retrieval event.'],['Memories left out','No recall update.\nTheir connections continue aging.','Cutoff, slot limit or token limit\nexcludes the memory.']],{y:183,h:368,widths:[285,475,380],size:23});
foot(s,'A preview or merely seeing the cue changes no existing memory or connection strength.');
note(s,'What changes after a retrieval',`Use pre-update scores for all ranking and route attribution. For the example eta=.2, delta_h=25 and h_max=200 ticks. These are illustrative starting constants. Every prompt-included memory receives g += tanh((t-last_recall)/(2 tau_m)) and last_recall=t. Its winning graph route learns only when G>M, with each qualifying edge updated once. A direct tie gets no graph update. Renewal starts from decayed transmission z, so it does not silently restore the old unaged weight. Alternate and unselected routes do not learn. A false reminder can reinforce its association, which is a failure mode to measure.`,[src.graph,design]);
}
// 16
{
const s=slide('Graph sources and allowed connections',{backup:true});
table(s,[['Connection','Meaning','Initial scope'],['Entity to episode','A current entity matches\nan episode’s entity tag.','Main condition: explicitly\nobserved entity cues.'],['Same tick','Two separately stored events\nshare a creation tick.','Optional. The writer normally\nstores one episode per turn.'],['Adjacent episodes','One stored episode follows another.','Optional. Immediately previous\nor next stored episode.'],['Memory through one entity hub','A direct source cues another episode\nthrough two mentions edges.','Optional. Fixed seed budget\nand no recursive spreading.']],{y:181,h:386,widths:[295,425,420],size:23});
foot(s,'Example: receiving food beside a pinwheel links the encounter to that object. The object did not cause the help.');
note(s,'Graph sources and allowed connections',`Use observed entity cues for the main comparisons. Later, direct memory sources can use frozen M values and a fixed seed cutoff/budget. That differs from the prototype's all-memory raw-similarity sourcing and must be controlled separately. A shared-hub route multiplies its two separately aging physical links; the historical prototype used a simplified receiving-edge rule. Never let graph-boosted candidates become new sources during the same pass, and exclude self-return paths. Same-tick all-pairs edges only exist if multiple records share a tick. The writer stores one combined episode per agent turn. Place annotations must be configured explicitly and activated only when observed. The main condition does not use temporal or memory-source routes.\n\nCurrent retrieval sources come from structured observation, message senders and inventory, plus configured visible places. Stored episode tags also include explicitly named intended action targets, such as trying to create a bell. That records an intention, not proof of a completed encounter or action. The main example uses an actually observed artifact, so it needs no change. Current IDs are normalized names, not a learned identity resolver. Memory-source routes are off in the main condition.`,[design,src.graph,src.agent]);
}
// 17
{
const s=slide('Time and the experimental controls',{backup:true});
table(s,[['Test','What varies','Same at each time gap'],['H1: associative recovery','Time gap, competing memories\nand graph access on or off','Identical memory state,\ncue and prompt budget.'],['H2: lasting accessibility','Time since the same recall event\nand connection learning on or off','Equal ordinary recalls.\nOnly one copy updates its link.'],['Believability in the simulation','Memory condition across\nrepeated simulation runs','Same generation settings\nand prompt budgets.']],{y:184,h:327,widths:[275,410,455],size:23});
text(s,'Time still causes decay. We vary the gap and compare at each gap.',70,549,1130,84,32,{bold:true,color:C.teal});
note(s,'Time and the experimental controls',`Matched does not mean holding time constant throughout an experiment. At a 20-tick gap compare the two copies at 20 ticks, then separately compare at 100 ticks, and so on. Similarly, H2 matches the target memory's g and full-recall timestamp because otherwise a direct-memory rehearsal effect could explain the difference. Connection histories are the intended difference.\n\nIn the chart both versions recall at tick 60. At tick 150 they have each gone 90 ticks without another recall. Their direct-only score is .044 in both, but graph-supported scores are .135 and .199 because only one connection was renewed. Controlled probes establish this mechanism. Complete simulation runs cannot keep the subsequent histories identical once the agents behave differently, so analyze those separately as whole-system effects.\n\nControlled replay uses matched ordinary memory states at each delay. This isolates mechanism effects but cannot model all later behavioral feedback. Separate full simulation runs let histories diverge. Changing only the final graph flag cannot erase earlier graph-induced recalls or plan contents. Report those full trajectories separately and label the fixed-graph chart as a mechanism illustration.`,[src.graph,ROOT+'/output/analysis/proposal-examples-v10.json']);
}
// 18
{
const s=slide('Expected limits and failure cases',{backup:true});
table(s,[['Failure case','What we expect or need to measure'],['No active associated cue','No support through that cue. Other routes may still work.'],['Several episodes share a cue','Graph scores can override differences in text relevance.'],['A cue accumulates many episodes','Crowding can push even a surviving link below the cutoff.'],['Repeated false reminder','An irrelevant association can strengthen and recur.'],['Aging of stored content','Access changes. Stored text stays intact.']],{y:182,h:365,widths:[470,670],size:24});
foot(s,'Cue-triggered retrieval is a modeling hypothesis. Clinical trauma and emotion require additional mechanisms.');
note(s,'Expected limits and failure cases',`An unused edge with finite h tends to zero, but a repeatedly qualifying link can remain accessible for a long time. That may be useful rehearsal or an undesirable loop. Track repeated and irrelevant retrieval rates, token crowding and competition with recent events. More recalls of everything would be a failure of selectivity. Max also ignores several weak cues agreeing; a bounded additive combination is a later sensitivity study. Fresh edge strength, beta and h must be calibrated together. The model can recover exact stored text even after a long delay, so it should not be presented as modeling reconstructive recall or emotional conditioning.\n\nThe audit reproduces a default-k=3 counterexample: the direct condition returns a scenario-relevant episode with cosine 1 and score .289218. Graph access raises it to .297374, but three scenario-irrelevant episodes rise to .300059, .302766 and .305494 and take all three slots. Every score increases while the useful memory disappears. A separate fixed-age fixture gives p=.216473 for one link at age 100 and p=.146816 for five memories sharing that hub, below .15. At beta=.5 and cutoff=.15, a hub with 167 retained memories cannot by itself produce a passing score even at w=1 and zero edge age. Other cues and direct retrieval can still work. These are mathematical/model limits, not empirical failure rates. They require crowding and distraction sweeps before interpreting long-run persistence.`,[design]);
}
// 19
{
const s=slide('Questions we should be ready to answer',{backup:true});
table(s,[['Question','Answer'],['Does the example prove vector search fails?','Similarity is positive. Ordinary decay\nputs the old encounter below the cutoff.'],['Is graph structure necessary here?','A graph or weighted lookup can store\nthe same cue-to-episode associations.'],['Does reinforcement prove better memory?','It raises support by design. Useful recall\nand believability still need evaluation.'],['What does selected mean?','Its text enters the prompt after the cutoff\nand the memory and token limits.'],['What can max lose?','When graph support dominates, differences\nin direct text relevance can stop affecting rank.']],{y:182,h:382,widths:[475,665],size:23});
note(s,'Questions we should be ready to answer',`The Mia–pinwheel example concerns a specific episode losing access under semantic similarity plus ordinary decay. Raw similarity is positive (.3702 in the hunger-aware fixture), and both retrieval methods initially include it. The association-only window is integer ticks 33–139 under the fixed defaults. A stronger encoder, richer query or slower ordinary forgetting can alter a comparison. We explicitly test a hunger-aware query in this illustration rather than leaving the agent's need out.

One-hop associations can be represented by an adjacency table or a graph. That equivalence preserves the relational information and does not show that associations lack benefit. The experiment studies independently aging, rehearsed associations, not the superiority of graph storage or multi-hop reasoning.

The cue activates an episode, not an approach reward. The action model may infer from Mia's earlier help and remark that she could be found there again. Its actual action must be evaluated. Max avoids summing overlapping contributions but can suppress semantic differences when the graph wins; retained-link crowding can suppress busy cues. Both remain model choices to test. No formula was changed for this presentation revision.`,[src.graph,src.agent,ROOT+'/output/analysis/pinwheel-walkthrough.json']);
}
// 20
{
const s=slide('Sources',{backup:true});
table(s,[['Source','Role in the project'],['Hou, Tamoto & Miyashita (2024)','Direct-memory decay and rehearsal'],['Honda et al., HAI 2025 proceedings','Embedding choice and memory-model context'],['Anderson & Reder, 1999','Associative competition and the fan effect'],['Paolo et al., TerraLingua, 2026','Persistent multi-agent simulation environment'],['APA Dictionary of Psychology','Associative-memory motivation'],['Kuroki et al., Shachi','Agent-believability evaluation context'],['Huang et al., EmotionBench','Secondary behavioral-evaluation context'],['Course proposal and design brief','Project goal, proposed equations and worked examples']],{y:165,h:447,widths:[500,640],size:22});
foot(s,'Full references appear in the speaker notes. The independent-pathway equations are our proposal.',633);
note(s,'Sources',`Hou: ${src.hou}\nHonda: ${src.honda}\nAssociative competition: ${fan}\nTerraLingua: ${src.terra}\nAPA: ${src.apa}\nShachi: ${src.shachi}\nEmotionBench: ${src.emotion}\nThe supplied course proposal and the independent associative retrieval design brief define our scope. The exact post-decay max, edge aging, reinforcement rule and numerical settings are proposed engineering choices, not equations validated by these references.`,[src.proposal,design,ROOT+'/output/analysis/independent-graph-walkthrough.json']);
}
await fs.mkdir(OUT,{recursive:true});
await fs.mkdir(BUILD,{recursive:true});
const candidate=path.join(BUILD,'candidate-v12.pptx');
await (await PresentationFile.exportPptx(p)).save(candidate);
await fs.writeFile(path.join(BUILD,'notes-v12.json'),JSON.stringify(notes,null,2));
if(process.argv.includes('--draft-only')){console.log(JSON.stringify({slides:p.slides.items.length,candidate,tableOwners,chartOwners}));process.exit(0);}
const final=path.join(OUT,'relational-memory-proposal-v12.pptx');
const result=await finalizePresentation({workspaceDir:ROOT,candidatePath:candidate,finalPath:final,pythonExecutable:PY,integrityValidatorPath:path.join(SKILL,'container_tools/inspect_presentation_package_integrity.py'),layoutValidatorPath:path.join(SKILL,'container_tools/inspect_presentation_layout_geometry.py'),layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit',...tableOwners.flatMap(n=>['--require-native-table-slide',String(n)])],requiredNativeTableOwnerSlides:tableOwners,requiredNativeChartOwnerSlides:chartOwners,materializeLiteralChartWorkbooks:true,fontPolicy:{basis:'reference',families:[FONT],referencePath:path.join(OUT,'archive/relational-memory-proposal-v11.pptx'),referenceSha256:'d79e3b63019c934a3e2b1569ed58ee67e878529ff7cd09bca195fd07469ef277'},verifyArtifactToolImport:true,receiptPath:path.join(BUILD,'validation-v12-delivery.json')});
console.log(JSON.stringify({slides:p.slides.items.length,final,sha:result.finalSha256,warnings:result.presentationLayout?.warnings}));
