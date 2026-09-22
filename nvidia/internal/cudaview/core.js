const listings = [];
let syncScroll = true, frame = 0, scrollSource = null, hoverLine = null;
function scheduleListings() {
  if (frame) return;
  frame = requestAnimationFrame(() => {
    frame = 0;
    if (scrollSource && syncScroll) {
      // Listings align on the source line at the top of the one scrolled;
      // where it names none, they keep the same relative position.
      const s = scrollSource, ratio = s.element.scrollTop / Math.max(1, s.element.scrollHeight - s.element.clientHeight), key = s.topSourceKey();
      for (const v of listings) if (v !== s && v.group === s.group) {
        const max = Math.max(0, v.element.scrollHeight - v.element.clientHeight), hit = key && v.indexes?.source.get(key);
        v.expectedScroll = Math.min(max, hit ? hit.values().next().value * 18 : ratio * max);
        v.element.scrollTop = v.expectedScroll;
      }
    }
    scrollSource = null;
    listings.forEach(v => v.render());
  });
}
function sourceKey(l) { return l.source_file && l.source_line ? l.source_file + ':' + l.source_line : ''; }
class Listing {
  constructor(element, group) {
    this.element = element; this.group = group; this.lines = []; this.selected = -1;
    element.tabIndex = 0; element.setAttribute('role','listbox');
    element.addEventListener('pointerleave',()=>{hoverLine=null;scheduleListings();});
    this.table = document.createElement('div'); this.table.className = 'code-table'; element.replaceChildren(this.table);
    element.addEventListener('scroll', () => {
      if (this.expectedScroll == null || Math.abs(element.scrollTop - this.expectedScroll) > 1) scrollSource = this;
      this.expectedScroll = null; scheduleListings();
    });
    element.addEventListener('keydown', e => {
      if (!['ArrowDown','ArrowUp'].includes(e.key)) return;
      e.preventDefault(); this.go(Math.max(0, Math.min(this.lines.length-1, this.selected + (e.key === 'ArrowDown' ? 1 : -1))));
    });
    new ResizeObserver(scheduleListings).observe(element); listings.push(this);
  }
  set(tier) {
    hoverLine = null;
    this.tier = tier; this.lines = tier.lines || []; this.selected = -1; this.indexes = {};
    let width = 0;
    for (const k of ['opcode','semantic_tag','offset','source']) this.indexes[k] = new Map();
    this.lines.forEach((l,i) => {
      width = Math.max(width,l.text.length);
      for (const k of Object.keys(this.indexes)) {
        const value = k === 'source' ? sourceKey(l) : l[k]; if (!value) continue;
        if (!this.indexes[k].has(value)) this.indexes[k].set(value,new Set());
        this.indexes[k].get(value).add(i);
      }
    });
    this.table.style.height = this.lines.length * 18 + 'px';
    this.table.style.width = Math.max(100,width*7+260) + 'px';
    this.element.scrollTop = 0; scheduleListings();
  }
  topSourceKey() {
    const top = Math.floor(this.element.scrollTop / 18);
    for (let i = top; i < Math.min(this.lines.length, top + 200); i++) { const k = sourceKey(this.lines[i]); if (k) return k; }
    return '';
  }
  go(i) {
    if (i < 0 || i >= this.lines.length) return;
    this.selected = i; this.element.scrollTop = i*18; inspectLine(this.lines[i],this.tier); scheduleListings();
  }
  render() {
    const start = Math.max(0,Math.floor(this.element.scrollTop/18)-50);
    const end = Math.min(this.lines.length,Math.ceil((this.element.scrollTop+this.element.clientHeight)/18)+50);
    const fragment = document.createDocumentFragment();
    for (let i=start;i<end;i++) {
      const l=this.lines[i], row=document.createElement('div'); row.className='code-row'; if(l.diff_type)row.classList.add('row-'+l.diff_type); row.style.top=i*18+'px';
      row.setAttribute('role','option'); row.setAttribute('aria-selected',i===this.selected);
      if (i===this.selected) row.classList.add('selected');
      if (hoverLine) for (const [k,cls] of [['opcode','highlight-match'],['semantic_tag','highlight-semtag'],['source','highlight-srcline']]) {
        const value=k==='source'?sourceKey(hoverLine):hoverLine[k];
        if (value && this.indexes[k].get(value)?.has(i)) row.classList.add(cls);
      }
      const number=document.createElement('div'); number.className='line-num'; number.textContent=l.num;
      const content=document.createElement('div'); content.className='line-content'; content.textContent=l.text;
      if (l.launch_kernel) {const launch=document.createElement('button');launch.className='badge-launch';launch.textContent='Launch: '+l.launch_kernel;launch.addEventListener('click',e=>{e.stopPropagation();onKernelChange(l.launch_kernel);});content.prepend(launch);}
      if (l.pressure) {const badge=document.createElement('span');badge.className='badge-reg';badge.textContent=l.pressure;content.prepend(badge);}
      content.title=[l.hover_doc,l.pressure_tooltip,l.stall_tooltip,l.remark_tooltip].filter(Boolean).join('\n');
      row.append(number,content);
      row.addEventListener('click',()=>{this.selected=i;inspectLine(l,this.tier);scheduleListings();});
      row.addEventListener('pointerenter',()=>{if(hoverLine!==l){hoverLine=l;scheduleListings();}});
      fragment.append(row);
    }
    this.table.replaceChildren(fragment);
  }
}
