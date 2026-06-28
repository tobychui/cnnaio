/* cnnaio API tester — builds requests, shows PNG/JSON results, and generates
   curl / fetch / jQuery snippets for the current form. Uses jQuery (provided). */
(function () {
  "use strict";

  // Endpoint catalogue. mode: how the request is sent.
  //   multipart  -> file upload (image + form fields)
  //   compare    -> JSON with two base64 images
  //   analyze    -> JSON with one base64 image + task list
  var ENDPOINTS = {
    "classification":  { path: "/v1/images/classifications", mode: "multipart", models: ["mobilenet-v2", "yolo11n-cls"], params: ["top_k"], preview: false },
    "detection":       { path: "/v1/images/detections", mode: "multipart", models: ["yolo11n", "nanodet-plus-m"], params: ["score_threshold", "nms_threshold", "max_results"], preview: true },
    "segmentation":    { path: "/v1/images/segmentations", mode: "multipart", models: ["yolo11n-seg"], params: ["score_threshold", "nms_threshold"], preview: true },
    "pose":            { path: "/v1/images/poses", mode: "multipart", models: ["yolo11n-pose"], params: ["score_threshold", "nms_threshold"], preview: true },
    "oriented (OBB)":  { path: "/v1/images/oriented", mode: "multipart", models: ["yolo11n-obb"], params: ["score_threshold", "nms_threshold"], preview: true },
    "face detection":  { path: "/v1/faces/detections", mode: "multipart", models: ["ultraface-rfb-320", "ultraface-slim-320"], params: ["score_threshold", "nms_threshold"], preview: true },
    "face landmarks":  { path: "/v1/faces/landmarks", mode: "multipart", models: ["pfld"], params: ["cropped"], preview: true },
    "face embedding":  { path: "/v1/faces/embeddings", mode: "multipart", models: ["mbv2facenet"], params: ["cropped"], preview: false },
    "gender":          { path: "/v1/faces/gender", mode: "multipart", models: ["gender-mbv2-0.35"], params: ["cropped"], preview: false },
    "face comparison": { path: "/v1/faces/comparisons", mode: "compare", models: ["mbv2facenet"], params: ["threshold"], preview: false },
    "analyze (multi)": { path: "/v1/vision/analyze", mode: "analyze", models: [], params: ["tasks"], preview: true },
  };
  var ANALYZE_TASKS = ["detect", "faces", "gender", "landmarks", "pose", "segment", "classify", "oriented"];

  var $ = window.jQuery;

  function baseURL() { return ($("#baseUrl").val() || "").replace(/\/+$/, ""); }
  function token() { return ($("#token").val() || "").trim(); }
  function ep() { return ENDPOINTS[$("#task").val()]; }

  // ---- form wiring ----

  function init() {
    $("#baseUrl").val(localStorage.getItem("cxn_base") || "");
    $("#token").val(localStorage.getItem("cxn_token") || "");
    $("#baseUrl,#token").on("input", function () {
      localStorage.setItem("cxn_base", $("#baseUrl").val());
      localStorage.setItem("cxn_token", $("#token").val());
      regen();
    });

    Object.keys(ENDPOINTS).forEach(function (k) { $("#task").append(new Option(k, k)); });
    ANALYZE_TASKS.forEach(function (t) {
      $("#tasksBox").append('<label class="ui checkbox row" style="margin-right:12px"><input type="checkbox" class="atask" value="' + t + '" ' + (["detect", "faces", "gender"].indexOf(t) >= 0 ? "checked" : "") + "> " + t + "</label>");
    });

    $("#task").on("change", onTask);
    $("#image").on("change", function () { preview(this, "#thumb"); regen(); });
    $("#imageB").on("change", function () { preview(this, "#thumbB"); regen(); });
    $("#model").on("change", regen);
    $(".ui.form").on("input change", "input,select", regen);
    $("#submit").on("click", submit);

    onTask();
  }

  function onTask() {
    var e = ep();
    // models
    $("#model").empty();
    e.models.forEach(function (m) { $("#model").append(new Option(m, m)); });

    // which fields are visible for this task
    var show = {};
    e.params.forEach(function (p) { show[p] = true; });
    show.image = true;
    show.model = e.models.length > 0;
    show.imageB = e.mode === "compare";
    show.render = e.preview;
    show.async = e.mode !== "compare"; // comparison has no async path here

    $("[data-param]").each(function () {
      var p = $(this).data("param");
      $(this).toggleClass("cxn-hidden", !show[p]);
    });
    regen();
  }

  function preview(input, sel) {
    var f = input.files && input.files[0];
    if (!f) { $(sel).hide(); return; }
    var r = new FileReader();
    r.onload = function () { $(sel).attr("src", r.result).show(); };
    r.readAsDataURL(f);
  }

  // ---- collect request parameters from the form ----

  function scalarParams(e) {
    var p = {};
    e.params.forEach(function (name) {
      if (["score_threshold", "nms_threshold", "top_k", "max_results", "threshold"].indexOf(name) >= 0) {
        var v = $("#" + name).val();
        if (v !== "" && v != null) p[name] = v;
      } else if (name === "cropped" && $("#cropped").is(":checked")) {
        p.cropped = "true";
      }
    });
    return p;
  }
  function analyzeTasks() {
    return $(".atask:checked").map(function () { return this.value; }).get();
  }
  function wantRender(e) { return e.preview && $("#render").is(":checked"); }
  function wantAsync(e) { return $("#async").is(":checked") && !$("[data-param=async]").hasClass("cxn-hidden"); }

  // ---- submit ----

  function submit() {
    var e = ep(), url = baseURL() + e.path;
    setStatus("Sending…", false);
    $("#resultImg").hide(); $("#resultJson").addClass("cxn-hidden");
    $("#submit").addClass("loading");

    var done = function (obj, code) {
      $("#submit").removeClass("loading");
      if (obj && obj.object === "job" && obj.status && obj.status !== "succeeded" && obj.status !== "failed") {
        setStatus("Job " + obj.id + " queued — polling…", false);
        pollJob(obj.id);
        return;
      }
      showResult(obj, code);
    };
    var fail = function (msg) { $("#submit").removeClass("loading"); setStatus("Error: " + msg, true); };

    if (e.mode === "multipart") {
      var f = $("#image")[0].files[0];
      if (!f) return fail("choose an image");
      var fd = new FormData();
      fd.append("image", f);
      if ($("#model").val()) fd.append("model", $("#model").val());
      var sp = scalarParams(e);
      Object.keys(sp).forEach(function (k) { fd.append(k, sp[k]); });
      if (wantRender(e)) fd.append("render", "true");
      if (wantAsync(e)) fd.append("async", "true");
      ajax(url, fd, null, done, fail);
    } else if (e.mode === "compare") {
      var fa = $("#image")[0].files[0], fb = $("#imageB")[0].files[0];
      if (!fa || !fb) return fail("choose both images");
      Promise.all([toDataURL(fa), toDataURL(fb)]).then(function (b) {
        var body = { image_a: b[0], image_b: b[1] };
        if ($("#threshold").val() !== "") body.threshold = parseFloat($("#threshold").val());
        ajax(url, null, body, done, fail);
      });
    } else { // analyze
      var fi = $("#image")[0].files[0];
      if (!fi) return fail("choose an image");
      toDataURL(fi).then(function (b64) {
        var body = { image: b64, tasks: analyzeTasks() };
        if (wantRender(e)) body.render = true;
        if (wantAsync(e)) body.async = true;
        ajax(url, null, body, done, fail);
      });
    }
  }

  function ajax(url, formData, jsonBody, done, fail) {
    var opt = { url: url, method: "POST", headers: {} };
    if (token()) opt.headers.Authorization = "Bearer " + token();
    if (formData) { opt.data = formData; opt.processData = false; opt.contentType = false; }
    else { opt.data = JSON.stringify(jsonBody); opt.contentType = "application/json"; }
    $.ajax(opt)
      .done(function (data, _s, xhr) { done(data, xhr.status); })
      .fail(function (xhr) {
        var msg = xhr.responseText || xhr.statusText;
        try { msg = JSON.parse(xhr.responseText).error.message; } catch (e) {}
        fail("HTTP " + xhr.status + " — " + msg);
      });
  }

  function pollJob(id) {
    var url = baseURL() + "/v1/jobs/" + id, tries = 0;
    var tick = function () {
      $.ajax({ url: url, headers: token() ? { Authorization: "Bearer " + token() } : {} })
        .done(function (job) {
          if (job.status === "succeeded") { setStatus("Job succeeded", false); showResult(job.result, 200); }
          else if (job.status === "failed") { setStatus("Job failed", true); showResult(job, 200); }
          else if (tries++ < 60) { setTimeout(tick, 1000); }
          else setStatus("Job still running after 60s", true);
        })
        .fail(function () { setStatus("Job poll failed", true); });
    };
    tick();
  }

  // ---- results ----

  function showResult(obj, code) {
    setStatus("HTTP " + (code || 200) + " · " + (obj && obj.object ? obj.object : ""), false);
    var png = findRendered(obj);
    if (png) { $("#resultImg").attr("src", png).show(); } else { $("#resultImg").hide(); }
    $("#resultJson").text(JSON.stringify(obj, replacer, 2)).removeClass("cxn-hidden");
  }

  function findRendered(o) {
    if (!o || typeof o !== "object") return null;
    if (o.rendered_image) return o.rendered_image;
    if (o.results) { for (var k in o.results) if (o.results[k].rendered_image) return o.results[k].rendered_image; }
    if (Array.isArray(o.data)) { for (var i = 0; i < o.data.length; i++) if (o.data[i] && o.data[i].rendered_image) return o.data[i].rendered_image; }
    return null;
  }

  // Truncate huge strings (data URIs / masks) and long arrays for readability.
  function replacer(key, val) {
    if (typeof val === "string" && val.length > 120) return val.slice(0, 48) + "…(" + val.length + " chars)";
    if (Array.isArray(val) && val.length > 24) return val.slice(0, 8).concat(["…" + val.length + " items"]);
    return val;
  }

  function setStatus(msg, isErr) {
    $("#status").text(msg).css("color", isErr ? "#c0392b" : "");
  }

  function toDataURL(file) {
    return new Promise(function (res) { var r = new FileReader(); r.onload = function () { res(r.result); }; r.readAsDataURL(file); });
  }

  // ---- code generation ----

  function regen() {
    var e = ep(), url = baseURL() + e.path, tok = token();
    $("#code-curl").text(genCurl(e, url, tok));
    $("#code-fetch").text(genFetch(e, url, tok));
    $("#code-jquery").text(genJquery(e, url, tok));
  }

  function fields(e) {
    // ordered [name,value] pairs for multipart/form
    var out = [];
    if ($("#model").val()) out.push(["model", $("#model").val()]);
    var sp = scalarParams(e);
    Object.keys(sp).forEach(function (k) { out.push([k, sp[k]]); });
    if (wantRender(e)) out.push(["render", "true"]);
    if (wantAsync(e)) out.push(["async", "true"]);
    return out;
  }

  function jsonBodyPreview(e) {
    if (e.mode === "compare") {
      var o = { image_a: "data:image/jpeg;base64,<BASE64_A>", image_b: "data:image/jpeg;base64,<BASE64_B>" };
      if ($("#threshold").val() !== "") o.threshold = parseFloat($("#threshold").val());
      return o;
    }
    var b = { image: "data:image/jpeg;base64,<BASE64_IMAGE>", tasks: analyzeTasks() };
    if (wantRender(e)) b.render = true;
    if (wantAsync(e)) b.async = true;
    return b;
  }

  function authHeaderCurl(tok) { return tok ? ' \\\n  -H "Authorization: Bearer ' + tok + '"' : ""; }

  function genCurl(e, url, tok) {
    if (e.mode === "multipart") {
      var lines = ['curl -X POST "' + url + '"' + authHeaderCurl(tok),
        '  -F "image=@/path/to/image.jpg"'];
      fields(e).forEach(function (kv) { lines.push('  -F "' + kv[0] + "=" + kv[1] + '"'); });
      return lines.join(" \\\n");
    }
    return 'curl -X POST "' + url + '"' + authHeaderCurl(tok) + ' \\\n' +
      '  -H "Content-Type: application/json"' + ' \\\n' +
      "  -d '" + JSON.stringify(jsonBodyPreview(e)) + "'";
  }

  function genFetch(e, url, tok) {
    var h = tok ? '{ "Authorization": "Bearer ' + tok + '" }' : "{}";
    if (e.mode === "multipart") {
      var s = "const fd = new FormData();\n";
      s += 'fd.append("image", fileInput.files[0]); // <input type=\"file\">\n';
      fields(e).forEach(function (kv) { s += 'fd.append("' + kv[0] + '", "' + kv[1] + '");\n'; });
      s += '\nconst res = await fetch("' + url + '", {\n  method: "POST",\n  headers: ' + h + ',\n  body: fd,\n});\nconst data = await res.json();\nconsole.log(data);';
      return s;
    }
    var hj = tok ? '{ "Authorization": "Bearer ' + tok + '", "Content-Type": "application/json" }'
      : '{ "Content-Type": "application/json" }';
    return "const body = " + JSON.stringify(jsonBodyPreview(e), null, 2) + ";\n\n" +
      'const res = await fetch("' + url + '", {\n  method: "POST",\n  headers: ' + hj + ',\n  body: JSON.stringify(body),\n});\nconst data = await res.json();\nconsole.log(data);';
  }

  function genJquery(e, url, tok) {
    var h = tok ? '{ Authorization: "Bearer ' + tok + '" }' : "{}";
    if (e.mode === "multipart") {
      var s = "const fd = new FormData();\n";
      s += 'fd.append("image", $("#file")[0].files[0]);\n';
      fields(e).forEach(function (kv) { s += 'fd.append("' + kv[0] + '", "' + kv[1] + '");\n'; });
      s += "\n$.ajax({\n  url: \"" + url + "\",\n  method: \"POST\",\n  headers: " + h + ",\n  data: fd,\n  processData: false,\n  contentType: false,\n}).done(function (data) { console.log(data); });";
      return s;
    }
    var hj = tok ? '{ Authorization: "Bearer ' + tok + '" }' : "{}";
    return "const body = " + JSON.stringify(jsonBodyPreview(e), null, 2) + ";\n\n" +
      "$.ajax({\n  url: \"" + url + "\",\n  method: \"POST\",\n  headers: " + hj + ",\n  contentType: \"application/json\",\n  data: JSON.stringify(body),\n}).done(function (data) { console.log(data); });";
  }

  $(init);
})();
