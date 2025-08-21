let enabled = false;

document.getElementById("toggle").addEventListener("click", async () => {
  enabled = !enabled;
  const btn = document.getElementById("toggle");

  if (enabled) {
    btn.textContent = "Disable";
    btn.classList.remove("off");
    btn.classList.add("on");
    // 🚀 later: send a message to backend (Python or WebAssembly model)
    console.log("Clear Comms ENABLED");
  } else {
    btn.textContent = "Enable";
    btn.classList.remove("on");
    btn.classList.add("off");
    console.log("Clear Comms DISABLED");
  }
});
