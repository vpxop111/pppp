// Example of how to integrate with the SVG generator API from the client side

async function generateSVG(prompt) {
  try {
    const response = await fetch('http://localhost:5000/api/generate-svg', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ prompt }),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! Status: ${response.status}`);
    }

    const data = await response.json();
    
    // Log the results
    console.log('Original prompt:', data.original_prompt);
    console.log('Enhanced prompt:', data.enhanced_prompt);
    console.log('SVG code:', data.svg_code);
    
    // You could render the SVG directly
    // document.getElementById('svg-container').innerHTML = data.svg_code;
    
    return data;
  } catch (error) {
    console.error('Error generating SVG:', error);
    throw error;
  }
}

// Example usage
// generateSVG('Draw a mountain landscape with a sunset')
//   .then(data => {
//     // Handle the response
//     console.log('SVG generation successful!');
//   })
//   .catch(error => {
//     console.error('SVG generation failed:', error);
//   }); 