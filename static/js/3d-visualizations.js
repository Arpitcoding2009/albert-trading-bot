// 3D Visualizations for Albert Trading Bot
let scene, camera, renderer;
let particles = [];
let animationFrame;

function init3D() {
    // Initialize Three.js scene
    scene = new THREE.Scene();
    camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
    renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });
    
    // Setup renderer
    const container = document.getElementById('hero-3d');
    renderer.setSize(container.offsetWidth, container.offsetHeight);
    container.appendChild(renderer.domElement);
    
    // Position camera
    camera.position.z = 5;
    
    // Create particle system
    createParticles();
    
    // Add window resize handler
    window.addEventListener('resize', onWindowResize, false);
    
    // Start animation
    animate();
}

function createParticles() {
    const geometry = new THREE.BufferGeometry();
    const vertices = [];
    const colors = [];
    
    // Create particles
    for (let i = 0; i < 1000; i++) {
        vertices.push(
            Math.random() * 10 - 5,
            Math.random() * 10 - 5,
            Math.random() * 10 - 5
        );
        
        // Add colors (blue to green gradient)
        colors.push(
            0.2 + Math.random() * 0.2,  // R
            0.5 + Math.random() * 0.3,  // G
            0.8 + Math.random() * 0.2   // B
        );
    }
    
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(vertices, 3));
    geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
    
    const material = new THREE.PointsMaterial({
        size: 0.05,
        vertexColors: true,
        transparent: true,
        opacity: 0.8
    });
    
    const points = new THREE.Points(geometry, material);
    scene.add(points);
    particles.push(points);
}

function animate() {
    animationFrame = requestAnimationFrame(animate);
    
    // Rotate particles
    particles.forEach(points => {
        points.rotation.x += 0.001;
        points.rotation.y += 0.002;
    });
    
    // Update particle positions based on market data
    updateParticlesWithMarketData();
    
    renderer.render(scene, camera);
}

function updateParticlesWithMarketData() {
    // This function will be called with real market data
    // For now, we'll just add some random movement
    particles.forEach(points => {
        const positions = points.geometry.attributes.position.array;
        
        for (let i = 0; i < positions.length; i += 3) {
            positions[i] += (Math.random() - 0.5) * 0.01;     // x
            positions[i + 1] += (Math.random() - 0.5) * 0.01; // y
            positions[i + 2] += (Math.random() - 0.5) * 0.01; // z
            
            // Keep particles within bounds
            if (Math.abs(positions[i]) > 5) positions[i] *= 0.95;
            if (Math.abs(positions[i + 1]) > 5) positions[i + 1] *= 0.95;
            if (Math.abs(positions[i + 2]) > 5) positions[i + 2] *= 0.95;
        }
        
        points.geometry.attributes.position.needsUpdate = true;
    });
}

function onWindowResize() {
    const container = document.getElementById('hero-3d');
    camera.aspect = container.offsetWidth / container.offsetHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(container.offsetWidth, container.offsetHeight);
}

function updateVisualizationWithTradeData(tradeData) {
    // This function will be called when new trade data arrives
    // Update particle colors or positions based on trade performance
    particles.forEach(points => {
        const colors = points.geometry.attributes.color.array;
        const positions = points.geometry.attributes.position.array;
        
        // Example: Change particle colors based on trade performance
        for (let i = 0; i < colors.length; i += 3) {
            if (tradeData.profitable) {
                colors[i + 1] += 0.1; // Increase green
            } else {
                colors[i] += 0.1; // Increase red
            }
            
            // Normalize colors
            colors[i] = Math.min(Math.max(colors[i], 0), 1);
            colors[i + 1] = Math.min(Math.max(colors[i + 1], 0), 1);
            colors[i + 2] = Math.min(Math.max(colors[i + 2], 0), 1);
        }
        
        points.geometry.attributes.color.needsUpdate = true;
    });
}

function cleanup3D() {
    if (animationFrame) {
        cancelAnimationFrame(animationFrame);
    }
    
    if (renderer) {
        renderer.dispose();
    }
    
    particles = [];
    scene = null;
    camera = null;
    renderer = null;
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', init3D);
// Cleanup when page is unloaded
window.addEventListener('unload', cleanup3D);
