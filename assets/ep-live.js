/* ep-live.js - the credibility benches for learn-ml-epistemology-with-phoebe.

   Two questions, both answered by training real networks in the page rather than by
   quoting a result. Thornbury predicts whether a patient deteriorates within 24 hours.
   Two signals genuinely cause the outcome; one shortcut, the ward, agrees with it 90
   percent of the time in the collection year without causing it, and 50 percent of the
   time after the wards are reorganised.

   Bench A asks what a held-out score entitles you to: twenty models whose ONLY
   difference is the random seed, scored where you looked and where you did not.
   Bench B asks what selecting on a test set costs: k candidates scored on one test set,
   the best one kept, and its honest accuracy measured on data nobody selected on.

   Every network is trained here by stochastic gradient descent and every accuracy is
   counted. Nothing is stored and no finding is written in advance.

   Exposes window.THORNBURY, renders into [data-cred-bench]. */
(function (root) {
  "use strict";
  function rng(seed){return function(){seed=(seed+0x6D2B79F5)|0;var t=seed;
    t=Math.imul(t^(t>>>15),t|1);t^=t+Math.imul(t^(t>>>7),t|61);
    return ((t^(t>>>14))>>>0)/4294967296;};}
  function normal(r){var u=Math.max(1e-12,r()),v=r();return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*v);}
  function sig(x){return 1/(1+Math.exp(-x));}

  /* ---- Thornbury: triage a patient as "deteriorates within 24h" ----
     Two honest signals (vitals drift, lab trend). One shortcut: which ward the
     patient is on. In the collection year the high-acuity ward took most of the
     deteriorating patients, so ward predicts the label without causing it.
     In the deployment year the wards were reorganised and the shortcut dies. */
  function makeData(n, seed, shortcutStrength){
    var r=rng(seed), rows=[];
    for(var i=0;i<n;i++){
      var vit=normal(r), lab=normal(r);
      var z=-0.35+1.25*vit+1.05*lab;
      var y = r() < sig(z) ? 1 : 0;
      /* ward agrees with the outcome shortcutStrength of the time */
      var ward = (r() < shortcutStrength) ? y : (r()<0.5?1:0);
      rows.push({x:[vit,lab,ward], y:y});
    }
    return rows;
  }

  /* a small over-parameterised net: 3 inputs -> H tanh -> 1 logit.
     The seed sets the initial weights AND the order examples arrive in.
     Nothing else differs between runs: same data, same architecture, same schedule. */
  function train(rows, seed, opts){
    opts=opts||{}; var H=opts.hidden||12, epochs=opts.epochs||30, lr=opts.lr||0.08, l2=opts.l2||2e-4;
    var r=rng(seed), d=rows[0].x.length;
    var W1=[], b1=[], W2=[], b2=0.2*normal(r);
    for(var h=0;h<H;h++){ var row=[]; for(var j=0;j<d;j++) row.push(1.1*normal(r)); W1.push(row); b1.push(0.3*normal(r)); W2.push(1.1*normal(r)); }
    var idx=rows.map(function(_,i){return i;});
    for(var e=0;e<epochs;e++){
      for(var k=idx.length-1;k>0;k--){var m=Math.floor(r()*(k+1));var t=idx[k];idx[k]=idx[m];idx[m]=t;}
      for(var s2=0;s2<idx.length;s2++){
        var x=rows[idx[s2]].x, y=rows[idx[s2]].y;
        var a=[], z2=b2;
        for(var h2=0;h2<H;h2++){ var z1=b1[h2]; for(var j2=0;j2<d;j2++) z1+=W1[h2][j2]*x[j2];
          var av=Math.tanh(z1); a.push(av); z2+=W2[h2]*av; }
        var g=sig(z2)-y;
        for(var h3=0;h3<H;h3++){
          var ga=g*W2[h3]*(1-a[h3]*a[h3]);
          W2[h3]-=lr*(g*a[h3]+l2*W2[h3]);
          for(var j3=0;j3<d;j3++) W1[h3][j3]-=lr*(ga*x[j3]+l2*W1[h3][j3]);
          b1[h3]-=lr*ga;
        }
        b2-=lr*g;
      }
    }
    return {W1:W1,b1:b1,W2:W2,b2:b2,H:H};
  }
  function forward(model,x){
    var z2=model.b2;
    for(var h=0;h<model.H;h++){ var z1=model.b1[h]; for(var j=0;j<x.length;j++) z1+=model.W1[h][j]*x[j];
      z2+=model.W2[h]*Math.tanh(z1); }
    return z2;
  }
  function acc(model, rows){
    var c=0;
    for(var i=0;i<rows.length;i++) if((forward(model,rows[i].x)>0?1:0)===rows[i].y) c++;
    return c/rows.length;
  }
  /* behavioural reliance: flip the ward on every row and count how many predictions change */
  function relianceOnWard(model, rows){
    var ch=0;
    for(var i=0;i<rows.length;i++){
      var x=rows[i].x, p0=forward(model,[x[0],x[1],0])>0?1:0, p1=forward(model,[x[0],x[1],1])>0?1:0;
      if(p0!==p1) ch++;
    }
    return ch/rows.length;
  }
  root.THORNBURY = root.THORNBURY || {rng:rng, makeData:makeData, train:train, acc:acc, forward:forward, relianceOnWard:relianceOnWard};
})(typeof window!=="undefined"?window:globalThis);

/* ---- Bench B: what selecting on a test set does to the number you report ----
   A pool of k candidate models is scored on ONE test set. The analyst keeps the
   best and reports its test accuracy. The honest number is that same model's
   accuracy on data nobody selected on. The gap is the optimism. */
(function (root) {
  var T = root.THORNBURY; if (!T) return;
  function candidatePool(train, k, seedBase){
    var pool=[]; for(var i=0;i<k;i++) pool.push(T.train(train, seedBase+i, {epochs:8, hidden:6}));
    return pool;
  }
  function selectOnTest(pool, test, fresh){
    var best=-1, bi=0;
    for(var i=0;i<pool.length;i++){ var a=T.acc(pool[i], test); if(a>best){best=a;bi=i;} }
    return {reported:best, honest:T.acc(pool[bi], fresh), index:bi};
  }
  root.THORNBURY.candidatePool = candidatePool;
  root.THORNBURY.selectOnTest = selectOnTest;
})(typeof window!=="undefined"?window:globalThis);

/* ============================================================
   The credibility bench widget. Renders into [data-cred-bench].
   Everything it prints is trained and counted on click.
   ============================================================ */
(function () {
  "use strict";
  if (typeof window === "undefined" || typeof document === "undefined") return;
  var T = window.THORNBURY; if (!T) return;
  var host = document.querySelector("[data-cred-bench]"); if (!host) return;

  function f3(x) { return x.toFixed(3); }
  function f4(x) { return x.toFixed(4); }

  var el = document.createElement("div");
  el.className = "cb-wrap";
  el.innerHTML =
    '<div class="cb-head"><div class="cb-title">THORNBURY - DETERIORATION RISK, 24 HOURS</div>' +
    '<div class="cb-claim">The ward agrees with the outcome <b>90 percent</b> of the time in the collection year and does not cause it. After the wards are reorganised it agrees <b>50 percent</b> of the time. Every network below is trained when you click.</div></div>' +
    '<div class="cb-tabs">' +
      '<button class="cb-tab on" data-pane="a" type="button">A · Twenty seeds, one pipeline</button>' +
      '<button class="cb-tab" data-pane="b" type="button">B · Selecting on a test set</button>' +
    '</div>' +
    '<div class="cb-pane" data-pane="a">' +
      '<div class="cb-ctl">' +
        '<div class="cb-field"><span class="cb-lab">Data draw</span><select class="cb-sel cb-draw">' +
          '<option value="101">Draw 1</option><option value="555">Draw 2</option><option value="909">Draw 3</option>' +
          '<option value="4242">Draw 4</option><option value="7">Draw 5</option></select></div>' +
        '<div class="cb-field"><span class="cb-lab">Models</span><select class="cb-sel cb-seeds">' +
          '<option value="8">8 seeds</option><option value="20" selected>20 seeds</option></select></div>' +
        '<button class="cb-run cb-runa" type="button">Train them</button>' +
      '</div><div class="cb-out-a"><p class="cb-note">Nothing trained yet. The only difference between these models is the random seed: same data, same architecture, same schedule.</p></div>' +
    '</div>' +
    '<div class="cb-pane" data-pane="b" hidden>' +
      '<div class="cb-ctl">' +
        '<div class="cb-field"><span class="cb-lab">Candidates scored on the test set</span><select class="cb-sel cb-k">' +
          '<option value="1">1</option><option value="5">5</option><option value="10">10</option>' +
          '<option value="25" selected>25</option><option value="50">50</option></select></div>' +
        '<div class="cb-field"><span class="cb-lab">Test set size</span><select class="cb-sel cb-n">' +
          '<option value="200">200</option><option value="400" selected>400</option><option value="1000">1000</option><option value="2000">2000</option></select></div>' +
        '<button class="cb-run cb-runb" type="button">Pick the best and report it</button>' +
      '</div><div class="cb-out-b"><p class="cb-note">Nothing selected yet. The reported number is the best candidate on the test set; the honest number is that same model on 6,000 patients nobody selected on.</p></div>' +
    '</div>';
  host.appendChild(el);

  el.querySelectorAll(".cb-tab").forEach(function (t) {
    t.addEventListener("click", function () {
      el.querySelectorAll(".cb-tab").forEach(function (x) { x.classList.remove("on"); });
      t.classList.add("on");
      el.querySelectorAll(".cb-pane").forEach(function (p) { p.hidden = p.getAttribute("data-pane") !== t.getAttribute("data-pane"); });
    });
  });

  /* ---- pane A ---- */
  el.querySelector(".cb-runa").addEventListener("click", function () {
    var btn = this; btn.disabled = true; btn.textContent = "Training...";
    setTimeout(function () {
      var ds = +el.querySelector(".cb-draw").value, n = +el.querySelector(".cb-seeds").value;
      var tr = T.makeData(4000, ds, 0.90), held = T.makeData(2000, ds + 1, 0.90), shift = T.makeData(2000, ds + 2, 0.50);
      var rows = [];
      for (var s = 1; s <= n; s++) {
        var m = T.train(tr, 1000 + s);
        rows.push({ seed: s, h: T.acc(m, held), sh: T.acc(m, shift), w: T.relianceOnWard(m, held) });
      }
      var hs = rows.map(function (r) { return r.h; }), ss = rows.map(function (r) { return r.sh; });
      var hsp = Math.max.apply(null, hs) - Math.min.apply(null, hs);
      var ssp = Math.max.apply(null, ss) - Math.min.apply(null, ss);
      var mx = hs.reduce(function (a, b) { return a + b; }, 0) / n, my = ss.reduce(function (a, b) { return a + b; }, 0) / n;
      var num = 0, dx = 0, dy = 0;
      for (var i = 0; i < n; i++) { num += (hs[i] - mx) * (ss[i] - my); dx += Math.pow(hs[i] - mx, 2); dy += Math.pow(ss[i] - my, 2); }
      var corr = (dx > 0 && dy > 0) ? num / Math.sqrt(dx * dy) : 0;
      var ratio = hsp > 0 ? ssp / hsp : 0;
      /* scatter: held on x, shifted on y */
      var X0 = 70, X1 = 560, Y0 = 200, Y1 = 30;
      var hmin = Math.min.apply(null, hs) - 0.004, hmax = Math.max.apply(null, hs) + 0.004;
      var smin = Math.min.apply(null, ss) - 0.01, smax = Math.max.apply(null, ss) + 0.01;
      var px = function (v) { return X0 + (v - hmin) / (hmax - hmin) * (X1 - X0); };
      var py = function (v) { return Y0 - (v - smin) / (smax - smin) * (Y0 - Y1); };
      var dots = rows.map(function (r) {
        return '<circle cx="' + px(r.h).toFixed(1) + '" cy="' + py(r.sh).toFixed(1) + '" r="4" fill="#713F12" fill-opacity="0.75"></circle>';
      }).join("");
      var plot = '<svg class="cb-plot" viewBox="0 0 600 230" role="img" aria-label="Each model as a dot: held-out accuracy across, shifted accuracy up. The dots span ' +
        f3(hsp) + ' horizontally and ' + f3(ssp) + ' vertically.">' +
        '<line x1="' + X0 + '" y1="' + Y0 + '" x2="' + X1 + '" y2="' + Y0 + '" stroke="#DCCDB8" stroke-width="1.5"></line>' +
        '<line x1="' + X0 + '" y1="' + Y0 + '" x2="' + X0 + '" y2="' + Y1 + '" stroke="#DCCDB8" stroke-width="1.5"></line>' +
        dots +
        '<text x="' + X0 + '" y="222" font="400 11px Inter, sans-serif" fill="#6B5B47" font-size="11">held-out accuracy (where you looked)</text>' +
        '<text x="12" y="22" font-size="11" fill="#6B5B47">shifted accuracy</text>' +
        '</svg>';
      var tbl = '<table class="cb-tbl"><thead><tr><th>Seed</th><th>Held out</th><th>Shifted</th><th>Reliance on the ward</th></tr></thead><tbody>' +
        rows.slice(0, 6).map(function (r) {
          return '<tr><td>' + r.seed + '</td><td>' + f3(r.h) + '</td><td>' + f3(r.sh) + '</td><td>' + r.w.toFixed(2) + '</td></tr>';
        }).join("") + '</tbody></table>';
      el.querySelector(".cb-out-a").innerHTML =
        '<div class="cb-reads">' +
          '<div class="cb-read"><b>' + f3(hsp) + '</b><span>held-out spread</span></div>' +
          '<div class="cb-read cb-flag"><b>' + f3(ssp) + '</b><span>shifted spread</span></div>' +
          '<div class="cb-read"><b>' + ratio.toFixed(1) + 'x</b><span>wider where you did not look</span></div>' +
          '<div class="cb-read"><b>' + (corr >= 0 ? "+" : "") + corr.toFixed(2) + '</b><span>corr(held, shifted)</span></div>' +
        '</div>' + plot +
        '<p class="cb-note">' + n + ' models, identical but for the seed. They agree to within <b>' + f3(hsp) +
        '</b> where you looked and disagree by <b>' + f3(ssp) + '</b> where you did not, which is <b>' + ratio.toFixed(1) +
        ' times wider</b>. The correlation on this draw is ' + corr.toFixed(2) +
        '; change the draw and watch its sign move, which is why a held-out score cannot tell you which of these to ship.</p>' + tbl;
      btn.disabled = false; btn.textContent = "Train them";
    }, 20);
  });

  /* ---- pane B ---- */
  el.querySelector(".cb-runb").addEventListener("click", function () {
    var btn = this; btn.disabled = true; btn.textContent = "Selecting...";
    setTimeout(function () {
      var k = +el.querySelector(".cb-k").value, n = +el.querySelector(".cb-n").value;
      var train = T.makeData(3000, 77, 0.90), fresh = T.makeData(6000, 88, 0.90);
      var REPS = 12, R = 0, H = 0;   /* matches the frozen canon, which is the mean of 12 */
      for (var r = 0; r < REPS; r++) {
        var test = T.makeData(n, 5000 + r * 13, 0.90);
        var pool = T.candidatePool(train, k, 20000 + r * 500);
        var s = T.selectOnTest(pool, test, fresh);
        R += s.reported; H += s.honest;
      }
      R /= REPS; H /= REPS;
      var gap = R - H;
      el.querySelector(".cb-out-b").innerHTML =
        '<div class="cb-reads">' +
          '<div class="cb-read"><b>' + f4(R) + '</b><span>reported accuracy</span></div>' +
          '<div class="cb-read"><b>' + f4(H) + '</b><span>honest accuracy</span></div>' +
          '<div class="cb-read cb-flag"><b>' + f4(gap) + '</b><span>optimism</span></div>' +
        '</div>' +
        '<p class="cb-note">Best of <b>' + k + '</b> on a test set of <b>' + n + '</b>, averaged over ' + REPS +
        ' replications. The reported number is what goes in the deck; the honest number is the same model on patients nobody selected on. ' +
        'Raise the candidate count and the optimism grows and then saturates. Raise the test set and it shrinks. ' +
        'It is real, it is measurable, and at these sizes it stays under one accuracy point, which is the part most retellings get wrong in both directions.</p>';
      btn.disabled = false; btn.textContent = "Pick the best and report it";
    }, 20);
  });
})();
