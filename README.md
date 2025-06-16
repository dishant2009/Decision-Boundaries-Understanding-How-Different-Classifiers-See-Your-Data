# Decision Boundaries: Understanding How Different Classifiers See Your Data

An interactive visualization that demonstrates how different machine learning classifiers create fundamentally different decision boundaries, debunking the common misconception that "all classifiers are basically the same."

![Decision Boundaries Demo](https://img.shields.io/badge/demo-interactive-blue)
![JavaScript](https://img.shields.io/badge/javascript-ES6-yellow)
![Plotly.js](https://img.shields.io/badge/plotly.js-v2.18.0-green)
![License](https://img.shields.io/badge/license-MIT-blue)

## Overview

This educational tool reveals a critical concept in machine learning: different algorithms make different assumptions about how to separate classes. A linear classifier can only draw straight lines, while others can create curves, circles, or even disconnected regions. Understanding these differences is crucial for selecting the right algorithm for your data.

## Live Demo

Try the visualization: [dishant2009.github.io/decision-boundaries-viz](https://dishant2009.github.io/decision-boundaries-viz)

## Key Learning Objectives

1. **Understand classifier limitations**: Why linear models fail on non-linear data
2. **Visualize decision boundaries**: See how each algorithm interprets the same data differently
3. **Explore parameter effects**: How hyperparameters change boundary shapes
4. **Compare algorithms**: Side-by-side comparison of six different classifiers
5. **Recognize patterns**: Which classifier works best for which data structure

## Features

### Six Classifiers Implemented

1. **Linear (Logistic Regression)**
   - Creates only straight line boundaries
   - Perfect for linearly separable data
   - Fails completely on XOR, circles, spirals
   - Most interpretable, least flexible

2. **Polynomial SVM**
   - Creates curved boundaries
   - Degree parameter controls complexity
   - Can handle moderate non-linearity
   - Risk of overfitting with high degree

3. **RBF (Radial Basis Function) SVM**
   - Can create any boundary shape
   - Even disconnected regions
   - Gamma parameter controls locality
   - Most flexible but prone to overfitting

4. **k-Nearest Neighbors (k-NN)**
   - Creates jagged, locally adaptive boundaries
   - No training phase (lazy learning)
   - k parameter controls smoothness
   - Sensitive to noise and scale

5. **Decision Tree**
   - Creates axis-aligned rectangular boundaries
   - Naturally handles multi-class
   - Max depth controls complexity
   - Interpretable but can overfit

6. **Neural Network (MLP)**
   - Flexible non-linear boundaries
   - Hidden layers add complexity
   - Can approximate any function
   - Requires more data and tuning

### Dataset Patterns

1. **Linearly Separable**: The ideal case for linear classifiers
2. **XOR Problem**: Classic example where linear fails
3. **Nested Circles**: One class completely surrounds another
4. **Two Moons**: Crescent shapes requiring flexible boundaries
5. **Spiral**: Ultimate test of classifier flexibility
6. **Mixed Complexity**: Multiple disconnected regions

### Interactive Controls

- **Dataset Selection**: Switch between 6 challenging patterns
- **Noise Level**: Add random noise to test robustness
- **Sample Size**: Control number of training points
- **Classifier Parameters**: 
  - Polynomial degree (1-5)
  - RBF gamma (0.1-10)
  - k-NN neighbors (1-20)
  - Tree depth (1-10)
  - Neural network layers and neurons
- **Visualization Options**:
  - Show/hide confidence regions
  - Show/hide support vectors
  - Animate training process
  - Compare all classifiers

## How to Run

### Option 1: Direct Browser (Recommended)
```bash
# Save as decision-boundaries.html
# Open in modern web browser
# No installation required!
```

### Option 2: Local Web Server
```bash
# Python 3
python -m http.server 8000

# Python 2
python -m SimpleHTTPServer 8000

# Node.js
npx http-server

# Navigate to http://localhost:8000/decision-boundaries.html
```

### Option 3: GitHub Pages
1. Fork this repository
2. Enable GitHub Pages in settings
3. Access at `https://dishant2009.github.io/decision-boundaries-viz`

## Understanding the Visualizations

### Main Plot (Left)
- **Red dots**: Class A training points
- **Blue dots**: Class B training points
- **Background**: Current dataset pattern

### Decision Boundary Plot (Right)
- **Yellow line**: Decision boundary (50% probability)
- **Colored regions**: Confidence levels (if enabled)
- **Contours**: Probability gradients

### Comparison View
- **6 subplots**: All classifiers on same data
- **Visual comparison**: See different interpretations
- **Pattern recognition**: Identify best classifier for data

## Why Different Classifiers Matter

### The XOR Problem
```
Class A: (0,0) and (1,1)
Class B: (0,1) and (1,0)
```
- **Linear**: Impossible to separate with straight line
- **Solution**: Need curved or disconnected boundaries
- **Historical importance**: Led to development of neural networks

### Real-World Implications

1. **Medical Diagnosis**
   - Linear: Simple risk factors (age + cholesterol)
   - Non-linear: Complex interactions between symptoms

2. **Customer Segmentation**
   - Linear: Basic demographic splits
   - RBF: Complex behavioral patterns

3. **Image Recognition**
   - Linear: Cannot handle pixel patterns
   - Neural Networks: Learn hierarchical features

4. **Fraud Detection**
   - k-NN: Detect local anomalies
   - Trees: Rule-based detection

## Technical Implementation

### Algorithms

All classifiers are implemented from scratch for educational transparency:

```javascript
// Linear Classifier (Simplified)
class LinearClassifier {
    fit(X, y) {
        // Gradient descent on logistic loss
        for (let iter = 0; iter < iterations; iter++) {
            const predictions = this.predict_proba(X);
            const errors = y - predictions;
            this.weights += learning_rate * X.T.dot(errors);
        }
    }
    
    predict_proba(X) {
        return sigmoid(X.dot(this.weights) + this.bias);
    }
}
```

### Decision Boundary Calculation

```javascript
// Create mesh grid
for (let x = -3; x <= 3; x += 0.1) {
    for (let y = -3; y <= 3; y += 0.1) {
        const probability = classifier.predict_proba([[x, y]]);
        // Plot probability as heatmap
    }
}
// Extract 0.5 probability contour as decision boundary
```

### Libraries Used

- **Plotly.js 2.18.0**: Interactive plotting and contours
- **Numeric.js 1.2.6**: Numerical computations
- **Vanilla JavaScript**: No framework dependencies

## Common Misconceptions Addressed

### 1. "More complex is always better"
**Reality**: Complex models overfit on simple data
**Demo**: Try RBF on linearly separable data with high gamma

### 2. "All classifiers find similar patterns"
**Reality**: Fundamentally different mathematical approaches
**Demo**: Compare all classifiers on XOR problem

### 3. "Linear models are obsolete"
**Reality**: Best choice for truly linear relationships
**Demo**: Linear classifier on linearly separable data

### 4. "Neural networks solve everything"
**Reality**: Require more data and careful tuning
**Demo**: Try neural network with few training points

### 5. "k-NN is too simple to be useful"
**Reality**: Excellent for local patterns
**Demo**: k-NN on spiral dataset

## Educational Usage

### For Students
1. Start with linearly separable data
2. Progress to XOR to understand linear limitations
3. Experiment with parameters on each dataset
4. Use comparison view to build intuition

### For Instructors
1. Demonstrate impossibility theorems (XOR)
2. Show overfitting with high complexity
3. Illustrate bias-variance tradeoff
4. Interactive exercises during lectures

### Learning Path
1. **Linear → XOR**: Understand linear limitations
2. **Circles → Polynomial**: Need for curved boundaries
3. **Moons → RBF**: Flexibility vs overfitting
4. **Spiral → Comparison**: No universal best classifier

## Performance Considerations

- Smooth up to 300 points per class
- Real-time updates for parameter changes
- Animation may lag with >500 total points
- Comparison view renders 6 classifiers simultaneously

## Browser Compatibility

- Chrome 60+ ✅
- Firefox 60+ ✅
- Safari 12+ ✅
- Edge 79+ ✅
- Mobile browsers (touch support)

## Extending the Visualization

### Adding New Classifiers
```javascript
class MyClassifier {
    fit(X, y) { /* training logic */ }
    predict_proba(X) { /* probability predictions */ }
    predict(X) { /* class predictions */ }
}
```

### Adding New Datasets
```javascript
function generateMyDataset(nSamples, noise) {
    // Return {X: [[x1,y1], ...], y: [0,1,0,...]}
}
```

### Custom Visualizations
- Modify `plotDecisionBoundary()` for different styles
- Add 3D visualization for 3-feature datasets
- Include training loss curves

## Troubleshooting

### Issue: Classifier not converging
- Reduce learning rate
- Increase iterations
- Normalize features

### Issue: Unexpected boundaries
- Check parameter ranges
- Verify data generation
- Look for class imbalance

### Issue: Slow performance
- Reduce sample size
- Disable animation
- Use Chrome for best performance

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## Future Enhancements

- [ ] Add more classifiers (Random Forest, Naive Bayes)
- [ ] 3D visualization for 3-feature datasets
- [ ] Export trained models
- [ ] Add cross-validation visualization
- [ ] Include feature importance plots
- [ ] Mobile-optimized version

## License

MIT License - feel free to use in your own projects!

## Acknowledgments

- Inspired by the classic XOR problem that stalled AI research
- Scikit-learn's classifier comparison examples
- Andrew Ng's visualization techniques in ML courses
- The broader ML education community

## Citation

If you use this in academic work, please cite:
```
@software{decision_boundaries_viz,
  author = {Dishant Digdarshi},
  title = {Decision Boundaries: Interactive Classifier Visualization},
  year = {2024},
  url = {https://github.com/dishant2009/decision-boundaries-viz}
}
```

## Contact

- GitHub: [@dishant2009]([[https://github.com/yourusername](https://github.com/dishant2009)](https://github.com/dishant2009))
- LinkedIn: [Dishant Digdarshi]([[https://linkedin.com/in/yourprofile](https://www.linkedin.com/in/dishant-digdarshi-24430a254)](https://www.linkedin.com/in/dishant-digdarshi-24430a254))
- Email: digdarshidishant@gmail.com

---

**Remember**: No classifier is universally best. The key is matching the algorithm's assumptions to your data's structure!
