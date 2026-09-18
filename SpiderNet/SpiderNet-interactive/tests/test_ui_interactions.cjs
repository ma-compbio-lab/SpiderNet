const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const path = require('node:path');
const base = path.join(__dirname, '../static/js');
function harness(module, exports = '') {
  const elements = new Map(), events = {}, requests = [];
  const element = id => {
    if (!elements.has(id)) elements.set(id, {id, value:'1', min:'',max:'',step:'', disabled:false, textContent:'',innerHTML:'',dataset:{},children:[],
      handlers:{}, addEventListener(k,f) {(this.handlers[k] ||= []).push(f);},
      classList:{add(){},remove(){},toggle(){}}, replaceChildren(...v){this.children=v;}, appendChild(v){this.children.push(v);},
      append(...v){this.children.push(...v);},querySelectorAll(){return [];},setAttribute(){},focus(){this.focused=true;}});
    return elements.get(id);
  };
  const context={console, performance, URLSearchParams, AbortController, ArrayBuffer, setTimeout, clearTimeout,
    document:{getElementById:element,createElement:()=>element('new'+Math.random()),querySelectorAll:()=>[],documentElement:{getAttribute:()=> 'light'}},
    window:{addEventListener(k,f){(events[k] ||= []).push(f);},spnValidateInputs:()=>true,spnRenderPlot(){}},
    Plotly:{purge(){},react(){}},fetch:(url,opts)=>{requests.push({url,opts}); return new Promise(()=>{});}};
  let source=fs.readFileSync(path.join(base,module+'.js'),'utf8');
  source=source.replace(/\}\)\(\);\s*$/,exports+'\n})();');
  vm.runInNewContext(source,context);
  return {context,element,events,requests};
}
(async()=>{
  for (const module of ['m1_basic','m2_subtype','m3_cascade','m4_perturb']) {
    const h=harness(module);
    const before=h.requests.length;
    for(const fn of h.events['spn:themechange']||[])fn();
    assert.equal(h.requests.length,before,module+' theme must not request analysis or feature tables');
  }
  console.log('PASS: all four theme handlers avoid analysis requests');
  {
    const h=harness('m1_basic','window.audit={refreshSliceBound};');
    const resolvers=[];
    h.context.fetch=()=>new Promise(resolve=>resolvers.push(resolve));
    const a=h.context.window.audit.refreshSliceBound();
    const b=h.context.window.audit.refreshSliceBound();
    const response=v=>({ok:true,json:async()=>({vmin:0,vmax:1,default:v,available_sender_celltypes:[],available_receiver_celltypes:[]})});
    resolvers[1](response(.2));await b;resolvers[0](response(.1));await a;
    assert.equal(h.element('m1-threshold').value,.2);
  }
  console.log('PASS: latest slice/MI threshold wins out-of-order responses');
  {
    const plots=[], listeners={};
    const genes=['A','B'], original={data:[{x:[1,2],y:[3,4],customdata:genes,marker:{color:['red','blue'],line:{color:'#111111'}}}],layout:{font:{color:'#111111'},xaxis:{range:[0,3],color:'#111111'}}};
    let mode='dark';
    const el={...original,_context:{}};
    const doc={documentElement:{getAttribute:()=>mode},querySelectorAll:()=>[el],getElementById:()=>({value:'-1',min:'0',max:'1',step:'',labels:[{textContent:'Threshold'}],focus(){}})};
    const context={ArrayBuffer,document:doc,window:{Plotly:{},addEventListener(k,f){listeners[k]=f;}},Plotly:{react:(target,data,layout)=>plots.push({target,data,layout})}};
    vm.runInNewContext(fs.readFileSync(path.join(base,'main.js'),'utf8').split('// END SHARED UI HELPERS')[0],context);
    listeners['spn:themechange']();
    assert.equal(plots[0].layout.font.color,'#ffffff');
    assert.equal(plots[0].data[0].marker.line.color,'#ffffff');
    assert.strictEqual(plots[0].data[0].customdata,genes);
    assert.deepEqual(plots[0].data[0].marker.color,['red','blue']);
    assert.equal(JSON.stringify(plots[0].layout.xaxis.range),'[0,3]');
    assert.equal(original.layout.font.color,'#111111');
    const status={};assert.equal(context.window.spnValidateInputs(['test'],status),false);
    assert.match(status.textContent,/Invalid Threshold/);
  }
  console.log('PASS: theme preserves coordinates, gene names, category colors and zoom; numeric validation rejects invalid values');
})().catch(e=>{console.error(e);process.exitCode=1;});
