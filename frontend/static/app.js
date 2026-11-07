const form = document.querySelector('#flow-form');
const error = document.querySelector('#error');
const result = document.querySelector('#result');
const empty = document.querySelector('#empty-result');
let revision = 0;
const showEmpty = () => { result.hidden = true; empty.hidden = false; error.hidden = true; };
const example = {PROTOCOL:6,TCP_FLAGS:24,L7_PROTO:7,FLOW_DURATION_MILLISECONDS:200,IN_BYTES:420,OUT_BYTES:320,IN_PKTS:5,OUT_PKTS:4};
document.querySelector('#example').addEventListener('click', () => {
  revision++;
  for (const [key,value] of Object.entries(example)) form.elements[key].value = value;
  showEmpty();
});
document.querySelector('#clear').addEventListener('click', () => { revision++; form.reset(); showEmpty(); form.elements.PROTOCOL.focus(); });
form.addEventListener('input', () => { revision++; showEmpty(); });
form.addEventListener('submit', async event => {
  event.preventDefault(); error.hidden = true;
  const requestRevision = revision;
  const button = form.querySelector('button[type=submit]'); const label = button.querySelector('.button-label');
  button.disabled = true; label.textContent = 'Analyzing…';
  try {
    const payload = Object.fromEntries([...new FormData(form)].map(([key,value]) => [key,Number(value)]));
    const response = await fetch('/api/predict', {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
    if (!response.ok) throw new Error(response.status === 422 ? 'Please check the input values and try again.' : 'The analysis service is unavailable. Please try again.');
    const data = await response.json();
    if (requestRevision !== revision) return;
    document.querySelector('#classification').textContent = data.label === 1 ? 'Attack-like' : 'Benign-like';
    document.querySelector('#interpretation').textContent = data.label === 1 ? 'The model classified this flow as attack traffic.' : 'The model classified this flow as benign traffic.';
    document.querySelector('#probability').textContent = `${(data.attack_probability * 100).toFixed(1)}%`;
    empty.hidden = true; result.hidden = false;
  } catch (e) { if (requestRevision !== revision) return; showEmpty(); error.textContent = e.message || 'Unable to reach the analysis service.'; error.hidden = false; }
  finally { button.disabled = false; label.textContent = 'Analyze flow'; }
});
(async () => {
  const body = document.querySelector('#results-table');
  try {
    const response = await fetch('/api/results'); if (!response.ok) throw Error();
    const data = await response.json(); body.replaceChildren();
    const method = data.method;
    document.querySelector('#method-summary').textContent = `A separate reproducibility run: up to ${Number(method.sample_size_per_dataset).toLocaleString()} sampled flows per dataset, stratified 80/20 split and fold-local scaling and SMOTE. These are not the original notebook’s full-run results.`;
    for (const [dataset,details] of Object.entries(data.datasets)) {
      for (const [name,metrics] of Object.entries(details.models)) {
        const row = document.createElement('tr');
        const values = [dataset,name,...['f1','recall','false_alarm_rate'].map(key => `${(metrics[key]*100).toFixed(1)}%`)];
        const labels = ['Dataset','Model','F1 score','Detection rate','False alarm rate'];
        values.forEach((value,index) => { const cell = document.createElement('td'); cell.textContent=value; cell.dataset.label=labels[index]; row.append(cell); });
        body.append(row);
      }
    }
  } catch { body.innerHTML = '<tr><td colspan="5">Benchmark results are unavailable.</td></tr>'; }
})();
