function hydrate(data) {
 for (const line of data.lines || []) {
  line.text=data.sources[line.source];line.hover_doc=data.sources[line.doc];line.pressure_tooltip=data.sources[line.pressure_doc];line.stall_tooltip=data.sources[line.stall_doc];line.remark_tooltip=data.sources[line.remark_doc];
 }
 for (const tiers of [data.tiers,...Object.values(data.kernel_tiers||{})])
  for (const tier of tiers||[]) tier.lines=(tier.ranges||[]).flatMap(([start,end])=>data.lines.slice(start,end));
}
hydrate(viewData);
let activeData=viewData;
let searchMatches=[],searchPosition=-1;
function onModuleChange(i) {
 activeData=i===0?viewData:viewData.modules[i].data;
 if (!activeData.tiers[0]?.lines) hydrate(activeData);
 const select=document.getElementById('kernelSelect');select.replaceChildren(new Option('All Kernels',''));
 Object.keys(activeData.kernel_tiers||{}).forEach(k=>select.add(new Option(k,k)));
 document.getElementById('analysisStatus').textContent=(activeData.diagnostics||[]).join(' ');
 onKernelChange('');
}
function inspectLine(l,t) { document.getElementById('inspector').textContent=t.file_name+':'+l.num+'\n'+l.text+'\n'+[l.kernel, l.opcode, l.predicate, l.operands, l.encoding, l.source_file ? l.source_file+':'+l.source_line : '',l.hover_doc,l.pressure_tooltip,l.stall_tooltip,l.remark_tooltip].filter(Boolean).join('\n'); }
function toggleSyncScroll() {syncScroll=!syncScroll;document.getElementById('syncScrollBtn').textContent='Sync Scroll: '+(syncScroll?'ON':'OFF');}
function showTab(i) {document.querySelectorAll('.tier-column').forEach((c,j)=>c.classList.toggle('tab-active',i===j));scheduleListings();}
function onKernelChange(name) {
  document.getElementById('kernelSelect').value=name;
  const tiers=activeData.kernel_tiers?.[name]||activeData.tiers;
  tiers.forEach((t,i)=>mainListings[i].set(t));
  document.getElementById('inspector').textContent='Select an instruction';
  searchMatches=[];searchPosition=-1;
}

function searchListings() {
 const q=document.getElementById('search').value;searchMatches=[];searchPosition=-1;
 for(const view of mainListings) view.lines.forEach((line,i)=>{if(q&&line.text.includes(q))searchMatches.push({view,i});});
 nextSearch(1);
}
function nextSearch(direction) {
 if(searchMatches.length){searchPosition=(searchPosition+direction+searchMatches.length)%searchMatches.length;const hit=searchMatches[searchPosition];hit.view.go(hit.i);showTab(mainListings.indexOf(hit.view));}
 document.getElementById('searchStatus').textContent=searchMatches.length?(searchPosition+1)+' of '+searchMatches.length+' matches':'0 matches';
}
function goToLine() {
  const value=document.getElementById('location').value.trim(),v=mainListings[Number(document.getElementById('listingSelect').value)];
  const i=value.startsWith('0x')?v.lines.findIndex(l=>l.offset&&parseInt(l.offset,16)===Number(value)):v.lines.findIndex(l=>l.num===Number(value));v.go(i);
  document.getElementById('searchStatus').textContent=i<0?'Location not found':'Line '+v.lines[i].num;
}
async function copyRange() {
  const v=mainListings[Number(document.getElementById('listingSelect').value)],from=Number(document.getElementById('rangeFrom').value),to=Number(document.getElementById('rangeTo').value);
  const text=v.lines.filter(l=>l.num>=from&&l.num<=to).map(l=>l.text).join('\n');
  try{await navigator.clipboard.writeText(text);document.getElementById('searchStatus').textContent='Copied';}
  catch{document.getElementById('inspector').textContent=text;document.getElementById('searchStatus').textContent='Clipboard unavailable; copy inspector text';}
}
const mainListings=viewData.tiers.map((t,i)=>new Listing(document.getElementById('code-scroll-'+i),'main'));
onKernelChange(viewData.active_kernel||'');
requestAnimationFrame(()=>requestAnimationFrame(()=>{document.documentElement.dataset.initialized='true';}));
