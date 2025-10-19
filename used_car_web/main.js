const start = document.querySelector('#start');
const introduction = document.querySelector('#introduction');
const collectData = document.querySelector('.collect_data');
const resultSection = document.getElementById('result');
const predictedPrice = document.getElementById('predicted_price');
const newPredictionButton = document.getElementById('new_prediction');

start.addEventListener('click', (event) => {
  // Code to start the prediction process
  event.preventDefault();
  introduction.style.display = 'none';
  collectData.style.display = 'flex';
});


const brandSelect = document.getElementById('brand');
const modelSelect = document.getElementById('model');
const startOption = document.getElementById('start_option');

// Use async/await for cleaner fetch logic and cache data after first load
let brandModelsCache = null;

async function loadBrandModels() {
  try {
    if (!brandModelsCache) {
      const response = await fetch("../brand_models.json");
      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
      brandModelsCache = await response.json();
    }

    brandSelect.addEventListener('change', function () {
      startOption.textContent = "Select a model";
      const selectedBrand = this.value;
      const models = brandModelsCache[selectedBrand] || [];
      // Always include the default option first
      modelSelect.innerHTML = `<option id="start_option" disabled selected value="">Select a model</option>` +
        models.map(model => `<option value="${model}">${model}</option>`).join('');
    });
  } catch (error) {
    console.error('Error loading JSON:', error);
  }
}

loadBrandModels();

const fuelTypeSelect = document.getElementById('fuel_type');
const transmissionSelect = document.getElementById('transmission');
const drivenInput = document.getElementById('driven');
const engineInput = document.getElementById('engine_capacity');
const ownerOrderSelect = document.getElementById('owners');
const yearInput = document.getElementById('year');
const HasInsuranceSelect = document.getElementById('has_insurance');
const spareKey = document.getElementById('spare_key');
const regNumuberInput = document.getElementById('registration_number');
const predict = document.getElementById('submit');

predict.addEventListener('click', async (event) => {
  event.preventDefault();

  if (!isValidIndianRegistration(regNumuberInput.value)) {
    alert("Invalid registration number format.");
    return;
  }

  const order = (ownerOrderSelect.value === "1" ? 'st' : (ownerOrderSelect.value === "2" ? 'nd' : 'th'));
  const data = {
    "brand": brandSelect.value,
    "model": modelSelect.value,
    "fuel_type": fuelTypeSelect.value,
    "transmission": transmissionSelect.value,
    "km_driven": drivenInput.value,
    "engine_capacity(CC)": engineInput.value,
    "ownership": `${ownerOrderSelect.value}${order} owner`,
    "make_year": yearInput.value,
    "has_insurance": HasInsuranceSelect.value,
    "spare_key": spareKey.value,
    "reg_number": regNumuberInput.value,
    "reg_year": 'unknown',
    "overall_cost": 0,
    "title": 'unknown'
  };

  try {
    const response = await fetch("http://localhost:8000/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ data })
    });
    if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
    const result = await response.json();
    predictedPrice.textContent = `₹${result.prediction.toLocaleString()}`;
    resultSection.classList.remove('hidden');
  } catch (error) {
    console.error("Prediction fetch error:", error);
  }
});

function isValidIndianRegistration(regNum) {
  // Typical format: 2 letters, 2 digits, 1-2 letters, 4 digits
  const regex = /^[A-Z]{2}\d{2}[A-Z]{1,2}\d{4}$/i;
  return regex.test(regNum.trim());
}


newPredictionButton.addEventListener('click', () => {
  // Reset form and UI for a new prediction
  document.querySelector('form').reset();
  resultSection.classList.add('hidden');
  collectData.style.display = 'flex';
});