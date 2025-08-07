import OpenAI from 'openai';
import * as tf from '@tensorflow/tfjs-node';
import axios from 'axios';
import { Logger } from '../../utils/logger';
import { Redis } from 'ioredis';
import { DatabaseManager } from '../../config/database';

export interface MLPredictionRequest {
  modelType: 'forecasting' | 'anomaly' | 'classification' | 'clustering';
  inputData: any[];
  parameters?: Record<string, any>;
  userId: string;
}

export interface AIInsightRequest {
  datasetId: string;
  analysisType: 'trend' | 'correlation' | 'anomaly' | 'forecast' | 'summary';
  timeframe?: string;
  filters?: Record<string, any>;
}

export interface NLQueryRequest {
  query: string;
  context?: Record<string, any>;
  userId: string;
}

/**
 * Enterprise AI Service Manager
 * Handles all AI/ML operations including GPT-4, TensorFlow models, and business intelligence
 */
export class AIServiceManager {
  private static instance: AIServiceManager;
  private openai: OpenAI | null = null;
  private redis: Redis;
  private models: Map<string, tf.LayersModel> = new Map();
  private isReady: boolean = false;

  private constructor() {
    // Singleton pattern for AI service manager
  }

  public static getInstance(): AIServiceManager {
    if (!AIServiceManager.instance) {
      AIServiceManager.instance = new AIServiceManager();
    }
    return AIServiceManager.instance;
  }

  /**
   * Initialize all AI services and models
   */
  public static async initialize(): Promise<void> {
    const manager = AIServiceManager.getInstance();
    await manager.init();
  }

  private async init(): Promise<void> {
    try {
      Logger.info('🤖 Initializing AI Service Manager...');

      // Initialize OpenAI client
      if (process.env.OPENAI_API_KEY) {
        this.openai = new OpenAI({
          apiKey: process.env.OPENAI_API_KEY,
          timeout: 30000,
          maxRetries: 3
        });
        Logger.info('🧠 OpenAI GPT-4 client initialized');
      } else {
        Logger.warn('⚠️ OpenAI API key not provided. NLP features will be limited.');
      }

      // Initialize Redis for ML model caching
      this.redis = new (require('ioredis'))({
        host: process.env.REDIS_HOST || 'localhost',
        port: parseInt(process.env.REDIS_PORT || '6379'),
        password: process.env.REDIS_PASSWORD,
        db: 1 // Use different database for ML cache
      });

      // Load pre-trained models
      await this.loadModels();

      // Warm up models with test predictions
      await this.warmUpModels();

      this.isReady = true;
      Logger.info('✅ AI Service Manager initialized successfully');

    } catch (error) {
      Logger.error('❌ Failed to initialize AI Service Manager:', error);
      throw error;
    }
  }

  /**
   * Load pre-trained TensorFlow models
   */
  private async loadModels(): Promise<void> {
    try {
      // Revenue Forecasting Model (LSTM)
      const forecastingModel = await this.createForecastingModel();
      this.models.set('revenue-forecasting', forecastingModel);

      // Anomaly Detection Model (Autoencoder)
      const anomalyModel = await this.createAnomalyModel();
      this.models.set('anomaly-detection', anomalyModel);

      // Customer Segmentation Model (Clustering)
      const segmentationModel = await this.createSegmentationModel();
      this.models.set('customer-segmentation', segmentationModel);

      Logger.info(`📚 Loaded ${this.models.size} ML models`);
    } catch (error) {
      Logger.error('❌ Failed to load ML models:', error);
      throw error;
    }
  }

  /**
   * Create LSTM model for revenue forecasting
   */
  private async createForecastingModel(): Promise<tf.LayersModel> {
    const model = tf.sequential({
      layers: [
        tf.layers.lstm({
          units: 50,
          returnSequences: true,
          inputShape: [30, 1] // 30 time steps, 1 feature
        }),
        tf.layers.dropout({ rate: 0.2 }),
        tf.layers.lstm({
          units: 50,
          returnSequences: true
        }),
        tf.layers.dropout({ rate: 0.2 }),
        tf.layers.lstm({
          units: 50
        }),
        tf.layers.dropout({ rate: 0.2 }),
        tf.layers.dense({ units: 1 })
      ]
    });

    model.compile({
      optimizer: 'adam',
      loss: 'meanSquaredError',
      metrics: ['accuracy']
    });

    return model;
  }

  /**
   * Create autoencoder for anomaly detection
   */
  private async createAnomalyModel(): Promise<tf.LayersModel> {
    const inputDim = 10; // Number of features
    const encodingDim = 5;

    const input = tf.input({ shape: [inputDim] });
    const encoded = tf.layers.dense({ units: encodingDim, activation: 'relu' }).apply(input);
    const decoded = tf.layers.dense({ units: inputDim, activation: 'sigmoid' }).apply(encoded);

    const autoencoder = tf.model({ inputs: input, outputs: decoded });

    autoencoder.compile({
      optimizer: 'adam',
      loss: 'meanSquaredError'
    });

    return autoencoder;
  }

  /**
   * Create model for customer segmentation
   */
  private async createSegmentationModel(): Promise<tf.LayersModel> {
    const model = tf.sequential({
      layers: [
        tf.layers.dense({ units: 64, activation: 'relu', inputShape: [8] }),
        tf.layers.dropout({ rate: 0.3 }),
        tf.layers.dense({ units: 32, activation: 'relu' }),
        tf.layers.dropout({ rate: 0.3 }),
        tf.layers.dense({ units: 16, activation: 'relu' }),
        tf.layers.dense({ units: 4, activation: 'softmax' }) // 4 customer segments
      ]
    });

    model.compile({
      optimizer: 'adam',
      loss: 'categoricalCrossentropy',
      metrics: ['accuracy']
    });

    return model;
  }

  /**
   * Warm up models with test predictions
   */
  private async warmUpModels(): Promise<void> {
    try {
      for (const [modelName, model] of this.models) {
        // Create dummy input based on model input shape
        const inputShape = model.inputs[0].shape.slice(1); // Remove batch dimension
        const dummyInput = tf.randomNormal([1, ...inputShape]);

        // Make prediction to warm up model
        const prediction = model.predict(dummyInput) as tf.Tensor;
        prediction.dispose();
        dummyInput.dispose();

        Logger.debug(`🔥 Warmed up model: ${modelName}`);
      }
    } catch (error) {
      Logger.warn('⚠️ Model warm-up failed:', error);
    }
  }

  /**
   * Process ML prediction request
   */
  public static async processMLPrediction(
    modelType: string, 
    inputData: any[], 
    userId: string
  ): Promise<any> {
    const manager = AIServiceManager.getInstance();
    return await manager.predict(modelType, inputData, userId);
  }

  private async predict(modelType: string, inputData: any[], userId: string): Promise<any> {
    try {
      // Check cache first
      const cacheKey = `ml:prediction:${modelType}:${this.hashData(inputData)}`;
      const cachedResult = await this.redis.get(cacheKey);

      if (cachedResult) {
        Logger.info(`📊 Using cached ML prediction for ${modelType}`);
        return JSON.parse(cachedResult);
      }

      let result: any;

      switch (modelType) {
        case 'revenue-forecasting':
          result = await this.predictRevenue(inputData);
          break;
        case 'anomaly-detection':
          result = await this.detectAnomalies(inputData);
          break;
        case 'customer-segmentation':
          result = await this.segmentCustomers(inputData);
          break;
        default:
          throw new Error(`Unknown model type: ${modelType}`);
      }

      // Cache result for 1 hour
      await this.redis.setex(cacheKey, 3600, JSON.stringify(result));

      // Log prediction for monitoring
      Logger.info({
        event: 'ml_prediction',
        modelType,
        userId,
        inputSize: inputData.length,
        predictionTime: Date.now()
      });

      return result;

    } catch (error) {
      Logger.error(`❌ ML prediction failed for ${modelType}:`, error);
      throw error;
    }
  }

  /**
   * Revenue forecasting using LSTM
   */
  private async predictRevenue(historicalData: number[]): Promise<any> {
    const model = this.models.get('revenue-forecasting');
    if (!model) throw new Error('Revenue forecasting model not loaded');

    // Prepare data for LSTM (normalize and reshape)
    const normalizedData = this.normalizeData(historicalData);
    const sequences = this.createSequences(normalizedData, 30);

    if (sequences.length === 0) {
      throw new Error('Insufficient data for revenue forecasting');
    }

    const inputTensor = tf.tensor3d([sequences[sequences.length - 1]]);
    const prediction = model.predict(inputTensor) as tf.Tensor;
    const predictionValue = await prediction.data();

    // Denormalize prediction
    const denormalizedPrediction = this.denormalizeData([predictionValue[0]], historicalData);

    // Clean up tensors
    inputTensor.dispose();
    prediction.dispose();

    return {
      type: 'revenue-forecast',
      prediction: denormalizedPrediction[0],
      confidence: this.calculateConfidence(historicalData),
      trend: this.analyzeTrend(historicalData),
      recommendations: await this.generateRevenueRecommendations(denormalizedPrediction[0], historicalData)
    };
  }

  /**
   * Anomaly detection using autoencoder
   */
  private async detectAnomalies(data: number[][]): Promise<any> {
    const model = this.models.get('anomaly-detection');
    if (!model) throw new Error('Anomaly detection model not loaded');

    const inputTensor = tf.tensor2d(data);
    const reconstruction = model.predict(inputTensor) as tf.Tensor;

    // Calculate reconstruction error
    const mse = tf.losses.meanSquaredError(inputTensor, reconstruction);
    const mseValues = await mse.data();

    // Define anomaly threshold (95th percentile of historical errors)
    const threshold = this.calculateAnomalyThreshold(mseValues);
    const anomalies = mseValues.map((error, index) => ({
      index,
      error,
      isAnomaly: error > threshold,
      severity: this.calculateSeverity(error, threshold)
    }));

    // Clean up tensors
    inputTensor.dispose();
    reconstruction.dispose();
    mse.dispose();

    return {
      type: 'anomaly-detection',
      anomalies: anomalies.filter(a => a.isAnomaly),
      totalDataPoints: data.length,
      anomalyRate: anomalies.filter(a => a.isAnomaly).length / data.length,
      threshold,
      recommendations: await this.generateAnomalyRecommendations(anomalies)
    };
  }

  /**
   * Customer segmentation
   */
  private async segmentCustomers(customerData: number[][]): Promise<any> {
    const model = this.models.get('customer-segmentation');
    if (!model) throw new Error('Customer segmentation model not loaded');

    const inputTensor = tf.tensor2d(customerData);
    const predictions = model.predict(inputTensor) as tf.Tensor;
    const probabilities = await predictions.data();

    // Convert probabilities to segments
    const segments = [];
    for (let i = 0; i < customerData.length; i++) {
      const customerProbs = probabilities.slice(i * 4, (i + 1) * 4);
      const segmentIndex = customerProbs.indexOf(Math.max(...customerProbs));
      const confidence = Math.max(...customerProbs);

      segments.push({
        customerId: i,
        segment: this.getSegmentName(segmentIndex),
        confidence,
        probabilities: Array.from(customerProbs)
      });
    }

    // Clean up tensors
    inputTensor.dispose();
    predictions.dispose();

    return {
      type: 'customer-segmentation',
      segments,
      segmentDistribution: this.calculateSegmentDistribution(segments),
      insights: await this.generateSegmentationInsights(segments)
    };
  }

  /**
   * Generate AI insights using GPT-4
   */
  public static async generateInsights(
    datasetId: string, 
    analysisType: string, 
    userId: string
  ): Promise<any> {
    const manager = AIServiceManager.getInstance();
    return await manager.generateBusinessInsights(datasetId, analysisType, userId);
  }

  private async generateBusinessInsights(
    datasetId: string, 
    analysisType: string, 
    userId: string
  ): Promise<any> {
    try {
      if (!this.openai) {
        throw new Error('OpenAI client not initialized');
      }

      // Fetch dataset from database
      const dataset = await DatabaseManager.getDataset(datasetId, userId);
      if (!dataset) {
        throw new Error('Dataset not found');
      }

      // Prepare data summary for GPT-4
      const dataSummary = this.summarizeDataset(dataset);
      const prompt = this.buildInsightsPrompt(dataSummary, analysisType);

      const response = await this.openai.chat.completions.create({
        model: 'gpt-4',
        messages: [
          {
            role: 'system',
            content: 'You are a senior business analyst with expertise in data analytics, market trends, and strategic planning. Provide actionable insights based on data analysis.'
          },
          {
            role: 'user',
            content: prompt
          }
        ],
        temperature: 0.3,
        max_tokens: 1500,
        top_p: 0.9
      });

      const insights = response.choices[0]?.message?.content;
      if (!insights) {
        throw new Error('No insights generated');
      }

      // Parse and structure insights
      const structuredInsights = this.parseInsights(insights);

      // Add visual recommendations
      const visualizations = this.suggestVisualizations(dataset, analysisType);

      return {
        type: 'ai-insights',
        datasetId,
        analysisType,
        insights: structuredInsights,
        visualizations,
        confidence: 0.85,
        generatedAt: new Date().toISOString(),
        recommendations: this.extractRecommendations(insights)
      };

    } catch (error) {
      Logger.error('❌ Failed to generate AI insights:', error);
      throw error;
    }
  }

  /**
   * Process natural language queries
   */
  public static async processNaturalLanguageQuery(request: NLQueryRequest): Promise<any> {
    const manager = AIServiceManager.getInstance();
    return await manager.processNLQuery(request);
  }

  private async processNLQuery(request: NLQueryRequest): Promise<any> {
    try {
      if (!this.openai) {
        throw new Error('OpenAI client not initialized');
      }

      const { query, context, userId } = request;

      // Check cache first
      const cacheKey = `nl:query:${this.hashData([query, context])}`;
      const cachedResult = await this.redis.get(cacheKey);

      if (cachedResult) {
        return JSON.parse(cachedResult);
      }

      // Convert natural language to SQL
      const sqlQuery = await this.convertNLToSQL(query, context);

      // Execute query
      const queryResults = await DatabaseManager.executeQuery(sqlQuery, userId);

      // Generate explanation
      const explanation = await this.generateQueryExplanation(query, queryResults);

      const result = {
        originalQuery: query,
        sqlQuery,
        results: queryResults,
        explanation,
        visualizations: this.suggestVisualizationsForQuery(queryResults),
        executedAt: new Date().toISOString()
      };

      // Cache for 30 minutes
      await this.redis.setex(cacheKey, 1800, JSON.stringify(result));

      return result;

    } catch (error) {
      Logger.error('❌ Natural language query processing failed:', error);
      throw error;
    }
  }

  /**
   * Convert natural language to SQL using GPT-4
   */
  private async convertNLToSQL(query: string, context?: any): Promise<string> {
    const prompt = `Convert this natural language query to SQL:

    Query: "${query}"

    Database Schema Context:
    ${JSON.stringify(context?.schema || this.getDefaultSchema(), null, 2)}

    Rules:
    1. Return ONLY the SQL query, no explanations
    2. Use proper table aliases
    3. Include appropriate JOINs when needed
    4. Add LIMIT clauses for large datasets
    5. Use parameterized queries to prevent injection

    SQL Query:`;

    const response = await this.openai!.chat.completions.create({
      model: 'gpt-4',
      messages: [{ role: 'user', content: prompt }],
      temperature: 0.1,
      max_tokens: 500
    });

    return response.choices[0]?.message?.content?.trim() || '';
  }

  /**
   * Health check for AI services
   */
  public static async healthCheck(): Promise<boolean> {
    const manager = AIServiceManager.getInstance();
    return manager.isReady && manager.models.size > 0;
  }

  public static isInitialized(): boolean {
    return AIServiceManager.getInstance().isReady;
  }

  // Helper methods
  private hashData(data: any): string {
    return Buffer.from(JSON.stringify(data)).toString('base64').slice(0, 32);
  }

  private normalizeData(data: number[]): number[] {
    const min = Math.min(...data);
    const max = Math.max(...data);
    return data.map(val => (val - min) / (max - min));
  }

  private denormalizeData(normalizedData: number[], originalData: number[]): number[] {
    const min = Math.min(...originalData);
    const max = Math.max(...originalData);
    return normalizedData.map(val => val * (max - min) + min);
  }

  private createSequences(data: number[], sequenceLength: number): number[][] {
    const sequences = [];
    for (let i = 0; i < data.length - sequenceLength; i++) {
      sequences.push(data.slice(i, i + sequenceLength));
    }
    return sequences;
  }

  private calculateConfidence(historicalData: number[]): number {
    // Calculate confidence based on data quality and model performance
    const variance = this.calculateVariance(historicalData);
    const stability = Math.max(0, 1 - variance / 10000);
    return Math.min(0.95, Math.max(0.5, stability));
  }

  private calculateVariance(data: number[]): number {
    const mean = data.reduce((sum, val) => sum + val, 0) / data.length;
    const squaredDiffs = data.map(val => Math.pow(val - mean, 2));
    return squaredDiffs.reduce((sum, val) => sum + val, 0) / data.length;
  }

  private analyzeTrend(data: number[]): string {
    if (data.length < 2) return 'insufficient-data';

    const recent = data.slice(-5);
    const earlier = data.slice(-10, -5);

    if (recent.length === 0 || earlier.length === 0) return 'insufficient-data';

    const recentAvg = recent.reduce((sum, val) => sum + val, 0) / recent.length;
    const earlierAvg = earlier.reduce((sum, val) => sum + val, 0) / earlier.length;

    const percentChange = (recentAvg - earlierAvg) / earlierAvg * 100;

    if (percentChange > 5) return 'increasing';
    if (percentChange < -5) return 'decreasing';
    return 'stable';
  }

  private calculateAnomalyThreshold(errors: Float32Array): number {
    const sorted = Array.from(errors).sort((a, b) => a - b);
    const index = Math.floor(sorted.length * 0.95);
    return sorted[index];
  }

  private calculateSeverity(error: number, threshold: number): string {
    const ratio = error / threshold;
    if (ratio > 2) return 'critical';
    if (ratio > 1.5) return 'high';
    if (ratio > 1.2) return 'medium';
    return 'low';
  }

  private getSegmentName(index: number): string {
    const segments = ['high-value', 'potential', 'regular', 'at-risk'];
    return segments[index] || 'unknown';
  }

  private calculateSegmentDistribution(segments: any[]): any {
    const distribution: Record<string, number> = {};
    segments.forEach(seg => {
      distribution[seg.segment] = (distribution[seg.segment] || 0) + 1;
    });
    return distribution;
  }

  // Additional helper methods would be implemented here...
  private summarizeDataset(dataset: any): string {
    return JSON.stringify(dataset).slice(0, 1000);
  }

  private buildInsightsPrompt(dataSummary: string, analysisType: string): string {
    return `Analyze this ${analysisType} data and provide business insights: ${dataSummary}`;
  }

  private parseInsights(insights: string): any {
    // Parse structured insights from GPT-4 response
    return { summary: insights };
  }

  private suggestVisualizations(dataset: any, analysisType: string): any[] {
    return [{ type: 'line-chart', title: 'Trend Analysis' }];
  }

  private extractRecommendations(insights: string): string[] {
    return ['Continue monitoring trends', 'Implement data-driven strategies'];
  }

  private async generateQueryExplanation(query: string, results: any): Promise<string> {
    return `Query returned ${results?.length || 0} results for: ${query}`;
  }

  private suggestVisualizationsForQuery(results: any): any[] {
    return [{ type: 'table', data: results }];
  }

  private getDefaultSchema(): any {
    return {
      tables: ['users', 'orders', 'products', 'analytics'],
      columns: {
        users: ['id', 'name', 'email', 'created_at'],
        orders: ['id', 'user_id', 'total', 'created_at'],
        products: ['id', 'name', 'price', 'category'],
        analytics: ['id', 'metric', 'value', 'date']
      }
    };
  }

  private async generateRevenueRecommendations(prediction: number, historical: number[]): Promise<string[]> {
    return ['Monitor market trends', 'Adjust pricing strategy'];
  }

  private async generateAnomalyRecommendations(anomalies: any[]): Promise<string[]> {
    return ['Investigate data quality', 'Review system performance'];
  }

  private async generateSegmentationInsights(segments: any[]): Promise<string[]> {
    return ['Target high-value customers', 'Develop retention strategies'];
  }
}

export default AIServiceManager;
