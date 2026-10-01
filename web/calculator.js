const $ = (id) => document.getElementById(id);

const inputs = {
  cda: $("cdaInput"),
  temperature: $("temperatureInput"),
  altitude: $("altitudeInput"),
  crr: $("crrInput"),
  efficiency: $("efficiencyInput"),
  mass: $("massInput"),
  speed: $("speedInput"),
  power: $("powerInput"),
};

const featureContent = {
  cda: {
    title: "CdA estimate",
    summary: "Your aerodynamic drag area is the frontal area multiplied by its drag coefficient.",
    body: "CdA describes how much aerodynamic drag your rider and bike create. The camera estimates frontal area from the silhouette and applies the selected drag coefficient.",
    formula: "CdA = frontal area (A) x drag coefficient (Cd)",
    assumption: "The camera-derived value is an estimate. Position, clothing, wind, and calibration quality all affect it.",
    value: (state) => state.cda,
    unit: "m²",
  },
  density: {
    title: "Air density",
    summary: "Colder air and lower altitude make the same position produce more drag.",
    body: "Air density is calculated from temperature and barometric pressure estimated from altitude. It is the air term in every aerodynamic power calculation.",
    formula: "rho = pressure / (R x temperature in kelvin)",
    assumption: "Pressure uses a standard-atmosphere approximation; local weather pressure can refine the result.",
    value: (state) => state.density,
    unit: "kg/m³",
  },
  aero: {
    title: "Aero power",
    summary: "The power required to push through the air rises with the cube of speed.",
    body: "Aero power isolates the work spent overcoming aerodynamic drag at the selected speed and air density.",
    formula: "P_aero = 0.5 x rho x CdA x v³",
    assumption: "This assumes still air and no gradient. A headwind increases apparent air speed.",
    value: (state) => state.aeroPower,
    unit: "W",
  },
  rolling: {
    title: "Rolling power",
    summary: "Tire losses are approximately proportional to mass, rolling resistance, and speed.",
    body: "Rolling power estimates the energy lost where the tires meet the road. Lower CRR tires or a lighter system reduce this part of the effort.",
    formula: "P_roll = CRR x mass x g x v",
    assumption: "The road is treated as flat, and CRR is treated as constant across the selected speed.",
    value: (state) => state.rollingPower,
    unit: "W",
  },
  wheel: {
    title: "Wheel power",
    summary: "Drivetrain efficiency converts rider power into power available at the wheel.",
    body: "Wheel power is the input power after drivetrain losses. It is the power available to overcome aero drag, rolling resistance, and any ignored gradient.",
    formula: "P_wheel = rider power x drivetrain efficiency",
    assumption: "Efficiency is entered as a percentage and is assumed constant for this estimate.",
    value: (state) => state.wheelPower,
    unit: "W",
  },
  inferred: {
    title: "CdA from power",
    summary: "Reverse the power equation to estimate the CdA implied by a speed and power pair.",
    body: "After subtracting rolling power from wheel power, the remaining power is assigned to aerodynamic drag. This is useful for comparing field-test efforts.",
    formula: "CdA = (P_wheel - P_roll) / (0.5 x rho x v³)",
    assumption: "Any gradient, wind, acceleration, bearing losses, or power-meter error will appear as CdA error.",
    value: (state) => state.inferredCda,
    unit: "m²",
  },
  speed: {
    title: "Speed from power",
    summary: "Find the steady speed supported by the selected rider power and CdA.",
    body: "The calculator searches for the speed where wheel power equals aerodynamic power plus rolling power. This turns a CdA estimate into a practical performance projection.",
    formula: "P_wheel = 0.5 x rho x CdA x v³ + CRR x m x g x v",
    assumption: "The solve assumes a flat course, still air, and a steady effort with no acceleration.",
    value: (state) => state.projectedSpeed,
    unit: "km/h",
  },
};

let latestState = null;

function number(id, fallback) {
  const value = Number.parseFloat(inputs[id].value);
  return Number.isFinite(value) ? value : fallback;
}

function calculate() {
  const cda = Math.max(0.001, number("cda", 0.2729));
  const temperature = number("temperature", 20);
  const altitude = number("altitude", 0);
  const crr = Math.max(0, number("crr", 0.00366));
  const efficiency = Math.max(0.01, number("efficiency", 96.5) / 100);
  const mass = Math.max(1, number("mass", 80));
  const speedKmh = Math.max(0.1, number("speed", 36));
  const riderPower = Math.max(0, number("power", 200));
  const speed = speedKmh / 3.6;
  const kelvin = Math.max(180, temperature + 273.15);
  const pressure = 101325 * Math.pow(Math.max(0.2, 1 - 2.25577e-5 * altitude), 5.25588);
  const density = pressure / (287.05 * kelvin);
  const rollingPower = crr * mass * 9.80665 * speed;
  const wheelPower = riderPower * efficiency;
  const aeroPower = 0.5 * density * cda * speed ** 3;
  const inferredCda = speed > 0 ? Math.max(0, wheelPower - rollingPower) / (0.5 * density * speed ** 3) : 0;
  let low = 0;
  let high = 100;
  for (let index = 0; index < 48; index += 1) {
    const candidate = (low + high) / 2;
    const required = 0.5 * density * cda * candidate ** 3 + crr * mass * 9.80665 * candidate;
    if (required > wheelPower) high = candidate;
    else low = candidate;
  }

  latestState = {
    cda, density, rollingPower, wheelPower, aeroPower, inferredCda,
    projectedSpeed: low * 3.6,
  };
  render(latestState);
}

function format(value, digits = 1) {
  return Number.isFinite(value) ? value.toFixed(digits) : "--";
}

function render(state) {
  $("densityValue").textContent = format(state.density, 3);
  $("aeroValue").textContent = format(state.aeroPower, 0);
  $("rollingValue").textContent = format(state.rollingPower, 0);
  $("wheelValue").textContent = format(state.wheelPower, 0);
  const selected = document.querySelector(".feature-link.active")?.dataset.feature || "cda";
  const feature = featureContent[selected];
  const digits = selected === "density" ? 3 : selected === "cda" || selected === "inferred" ? 4 : selected === "speed" ? 1 : 0;
  const value = feature.value(state);
  $("resultTitle").textContent = feature.title;
  $("resultValue").innerHTML = `${format(value, digits)} <small>${feature.unit}</small>`;
  $("resultSummary").textContent = feature.summary;
  $("explainTitle").textContent = feature.title;
  $("explainBody").textContent = feature.body;
  $("formulaText").textContent = feature.formula;
  $("assumptionText").textContent = feature.assumption;
  $("resultTrackFill").style.width = `${Math.min(100, Math.max(9, (Math.abs(value) / (feature.unit === "W" ? 400 : feature.unit === "km/h" ? 60 : 1)) * 100))}%`;
}

document.querySelectorAll(".feature-link").forEach((button) => {
  button.addEventListener("click", () => {
    document.querySelectorAll(".feature-link").forEach((item) => item.classList.remove("active"));
    button.classList.add("active");
    render(latestState);
  });
});
Object.values(inputs).forEach((input) => input.addEventListener("input", calculate));

$("useLatestCda").addEventListener("click", () => {
  const saved = Number.parseFloat(localStorage.getItem("aerott.latestCda"));
  if (!Number.isFinite(saved) || saved <= 0) return;
  inputs.cda.value = saved.toFixed(4);
  $("cdaSource").textContent = "Latest camera CdA";
  calculate();
});

const savedCda = Number.parseFloat(localStorage.getItem("aerott.latestCda"));
if (Number.isFinite(savedCda) && savedCda > 0) {
  inputs.cda.value = savedCda.toFixed(4);
  $("cdaSource").textContent = "Latest camera CdA available";
}
calculate();
