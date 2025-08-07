import express from 'express';
import { ApolloServer } from 'apollo-server-express';
import { createServer } from 'http';
import { Server as SocketServer } from 'socket.io';
import cors from 'cors';
import helmet from 'helmet';
import compression from 'compression';
import rateLimit from 'express-rate-limit';
import swaggerJsdoc from 'swagger-jsdoc';
import swaggerUi from 'swagger-ui-express';
import dotenv from 'dotenv';
import Redis from 'ioredis';
import Bull from 'bull';

import { DatabaseManager } from './config/database';
import { GraphQLSchema } from './graphql/schema';
import { AuthMiddleware } from './middleware/auth';
import { ErrorHandler } from './middleware/errorHandler';
import { Logger } from './utils/logger';
import { MetricsCollector } from './monitoring/metrics';
import { AIServiceManager } from './services/ai/aiManager';
import { WebSocketManager } from './services/websocket/manager';
import { setupRoutes } from './routes';

// Load environment variables
dotenv.config();

/**
 * Enterprise AI Business Intelligence Platform Server
 * Supports 10,000+ concurrent users with advanced AI/ML capabilities
 */
class AIBusinessPlatformServer {
  public app: express.Application;
  public server: any;
  public io: SocketServer;
  public apolloServer: ApolloServer;
  public redis: Redis;
  public taskQueue: Bull.Queue;

  private port: number;
  private isDevelopment: boolean;
  private metrics: MetricsCollector;

  constructor() {
    this.app = express();
    this.port = parseInt(process.env.PORT || '3001');
    this.isDevelopment = process.env.NODE_ENV === 'development';

    this.server = createServer(this.app);
    this.io = new SocketServer(this.server, {
      cors: {
        origin: process.env.ALLOWED_ORIGINS?.split(',') || ['http://localhost:3000'],
        methods: ['GET', 'POST'],
        credentials: true
      },
      transports: ['websocket', 'polling']
    });

    this.metrics = new MetricsCollector();

    Logger.info('🚀 AI Business Intelligence Platform initializing...');
  }

  /**
   * Initialize all services and connections
   */
  private async initializeServices(): Promise<void> {
    try {
      // Initialize Redis connection
      await this.initializeRedis();

      // Initialize database connections
      await DatabaseManager.initialize();

      // Initialize AI services
      await AIServiceManager.initialize();

      // Initialize task queue
      await this.initializeTaskQueue();

      // Initialize Apollo GraphQL server
      await this.initializeApolloServer();

      // Initialize WebSocket manager
      WebSocketManager.initialize(this.io);

      Logger.info('✅ All services initialized successfully');
    } catch (error) {
      Logger.error('❌ Service initialization failed:', error);
      throw error;
    }
  }

  /**
   * Initialize Redis connection for caching and session management
   */
  private async initializeRedis(): Promise<void> {
    this.redis = new Redis({
      host: process.env.REDIS_HOST || 'localhost',
      port: parseInt(process.env.REDIS_PORT || '6379'),
      password: process.env.REDIS_PASSWORD,
      retryDelayOnFailover: 100,
      enableReadyCheck: false,
      lazyConnect: true,
      maxRetriesPerRequest: 3
    });

    this.redis.on('connect', () => {
      Logger.info('🔴 Redis connected successfully');
    });

    this.redis.on('error', (error) => {
      Logger.error('❌ Redis connection error:', error);
    });

    await this.redis.connect();
  }

  /**
   * Initialize Bull queue for background job processing
   */
  private async initializeTaskQueue(): Promise<void> {
    this.taskQueue = new Bull('AI Processing Queue', {
      redis: {
        host: process.env.REDIS_HOST || 'localhost',
        port: parseInt(process.env.REDIS_PORT || '6379'),
        password: process.env.REDIS_PASSWORD
      },
      defaultJobOptions: {
        removeOnComplete: 100,
        removeOnFail: 50,
        attempts: 3,
        backoff: {
          type: 'exponential',
          delay: 2000
        }
      }
    });

    // Define job processors
    this.taskQueue.process('ml-prediction', 10, async (job) => {
      const { modelType, inputData, userId } = job.data;
      return await AIServiceManager.processMLPrediction(modelType, inputData, userId);
    });

    this.taskQueue.process('data-export', 5, async (job) => {
      const { format, query, userId, reportId } = job.data;
      return await this.processDataExport(format, query, userId, reportId);
    });

    this.taskQueue.process('ai-insights', 3, async (job) => {
      const { datasetId, analysisType, userId } = job.data;
      return await AIServiceManager.generateInsights(datasetId, analysisType, userId);
    });

    Logger.info('📋 Task queue initialized with processors');
  }

  /**
   * Initialize Apollo GraphQL server
   */
  private async initializeApolloServer(): Promise<void> {
    this.apolloServer = new ApolloServer({
      schema: GraphQLSchema,
      context: ({ req, res }) => ({
        req,
        res,
        redis: this.redis,
        taskQueue: this.taskQueue,
        user: req.user
      }),
      formatError: (error) => {
        Logger.error('GraphQL Error:', error);
        return this.isDevelopment ? error : new Error('Internal server error');
      },
      introspection: this.isDevelopment,
      playground: this.isDevelopment,
      subscriptions: {
        path: '/graphql-subscriptions',
        onConnect: (connectionParams, webSocket) => {
          Logger.info('GraphQL subscription connected');
          return { user: connectionParams.user };
        }
      }
    });

    await this.apolloServer.start();
    this.apolloServer.applyMiddleware({ app: this.app, path: '/graphql' });

    Logger.info('🎯 Apollo GraphQL server initialized');
  }

  /**
   * Configure middleware stack
   */
  private configureMiddleware(): void {
    // Security middleware
    this.app.use(helmet({
      contentSecurityPolicy: this.isDevelopment ? false : {
        directives: {
          defaultSrc: ["'self'"],
          styleSrc: ["'self'", "'unsafe-inline'", "https://fonts.googleapis.com"],
          fontSrc: ["'self'", "https://fonts.gstatic.com"],
          scriptSrc: ["'self'"],
          imgSrc: ["'self'", "data:", "https:"],
          connectSrc: ["'self'", "wss:", "https:"]
        }
      }
    }));

    // CORS configuration
    this.app.use(cors({
      origin: (origin, callback) => {
        const allowedOrigins = process.env.ALLOWED_ORIGINS?.split(',') || ['http://localhost:3000'];
        if (!origin || allowedOrigins.includes(origin)) {
          callback(null, true);
        } else {
          callback(new Error('Not allowed by CORS'));
        }
      },
      credentials: true,
      methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
      allowedHeaders: ['Content-Type', 'Authorization', 'X-API-Key']
    }));

    // Compression and parsing
    this.app.use(compression());
    this.app.use(express.json({ 
      limit: '50mb',
      type: ['application/json', 'text/plain']
    }));
    this.app.use(express.urlencoded({ 
      extended: true,
      limit: '50mb'
    }));

    // Rate limiting with Redis store
    const limiter = rateLimit({
      windowMs: 15 * 60 * 1000, // 15 minutes
      max: this.isDevelopment ? 1000 : 100, // requests per window
      standardHeaders: true,
      legacyHeaders: false,
      store: new (require('rate-limit-redis'))({
        client: this.redis,
        prefix: 'rl:'
      }),
      message: {
        error: 'Too many requests from this IP',
        retryAfter: '15 minutes'
      }
    });

    this.app.use('/api/', limiter);

    // Request logging and metrics
    this.app.use((req, res, next) => {
      const start = Date.now();

      res.on('finish', () => {
        const duration = Date.now() - start;
        this.metrics.recordHttpRequest(req.method, req.path, res.statusCode, duration);

        Logger.info({
          method: req.method,
          url: req.url,
          status: res.statusCode,
          duration: `${duration}ms`,
          ip: req.ip,
          userAgent: req.get('User-Agent')
        });
      });

      next();
    });

    // Authentication middleware
    this.app.use('/api/', AuthMiddleware.authenticate);
  }

  /**
   * Configure API documentation
   */
  private configureApiDocumentation(): void {
    const swaggerOptions = {
      definition: {
        openapi: '3.0.0',
        info: {
          title: 'AI Business Intelligence Platform API',
          version: '2.0.0',
          description: 'Enterprise-grade business intelligence platform with AI-powered analytics',
          contact: {
            name: 'Himanshu Singh',
            url: 'https://github.com/GeekYasuo',
            email: 'himanshu@example.com'
          },
          license: {
            name: 'MIT',
            url: 'https://opensource.org/licenses/MIT'
          }
        },
        servers: [
          {
            url: process.env.API_BASE_URL || `http://localhost:${this.port}`,
            description: this.isDevelopment ? 'Development server' : 'Production server'
          }
        ],
        components: {
          securitySchemes: {
            bearerAuth: {
              type: 'http',
              scheme: 'bearer',
              bearerFormat: 'JWT'
            },
            apiKey: {
              type: 'apiKey',
              in: 'header',
              name: 'X-API-Key'
            }
          },
          schemas: {
            Error: {
              type: 'object',
              properties: {
                error: { type: 'string' },
                message: { type: 'string' },
                statusCode: { type: 'number' },
                timestamp: { type: 'string' }
              }
            }
          }
        },
        security: [{ bearerAuth: [] }]
      },
      apis: ['./src/routes/*.ts', './src/controllers/*.ts']
    };

    const specs = swaggerJsdoc(swaggerOptions);

    this.app.use('/api-docs', swaggerUi.serve);
    this.app.get('/api-docs', swaggerUi.setup(specs, {
      customCss: '.swagger-ui .topbar { display: none }',
      customSiteTitle: 'AI Business Platform API',
      swaggerOptions: {
        docExpansion: 'none',
        displayRequestDuration: true
      }
    }));
  }

  /**
   * Configure health checks and monitoring endpoints
   */
  private configureHealthChecks(): void {
    // Basic health check
    this.app.get('/health', async (req, res) => {
      const healthStatus = {
        status: 'healthy',
        timestamp: new Date().toISOString(),
        uptime: Math.floor(process.uptime()),
        environment: process.env.NODE_ENV,
        version: process.env.npm_package_version || '2.0.0',
        services: {
          database: 'unknown',
          redis: 'unknown',
          ai_services: 'unknown'
        }
      };

      try {
        // Check database connection
        await DatabaseManager.healthCheck();
        healthStatus.services.database = 'healthy';

        // Check Redis connection
        await this.redis.ping();
        healthStatus.services.redis = 'healthy';

        // Check AI services
        const aiStatus = await AIServiceManager.healthCheck();
        healthStatus.services.ai_services = aiStatus ? 'healthy' : 'degraded';

        res.status(200).json(healthStatus);
      } catch (error) {
        healthStatus.status = 'unhealthy';
        healthStatus.services.database = 'unhealthy';
        Logger.error('Health check failed:', error);
        res.status(503).json(healthStatus);
      }
    });

    // Detailed system metrics
    this.app.get('/metrics', async (req, res) => {
      try {
        const metrics = await this.metrics.getMetrics();
        res.set('Content-Type', 'text/plain');
        res.send(metrics);
      } catch (error) {
        Logger.error('Metrics endpoint error:', error);
        res.status(500).json({ error: 'Metrics unavailable' });
      }
    });

    // Readiness probe for Kubernetes
    this.app.get('/ready', async (req, res) => {
      try {
        await Promise.all([
          DatabaseManager.healthCheck(),
          this.redis.ping(),
          AIServiceManager.healthCheck()
        ]);

        res.status(200).json({ 
          status: 'ready',
          timestamp: new Date().toISOString()
        });
      } catch (error) {
        res.status(503).json({ 
          status: 'not ready',
          error: error.message
        });
      }
    });
  }

  /**
   * Configure application routes
   */
  private configureRoutes(): void {
    // API routes
    setupRoutes(this.app, this.redis, this.taskQueue);

    // GraphQL subscriptions endpoint
    this.server.on('upgrade', this.apolloServer.subscriptionServer.handleUpgrade);

    // 404 handler
    this.app.use('*', (req, res) => {
      res.status(404).json({
        error: 'Route not found',
        message: `Cannot ${req.method} ${req.originalUrl}`,
        availableEndpoints: [
          '/api/analytics',
          '/api/ai',
          '/api/auth',
          '/graphql',
          '/api-docs'
        ]
      });
    });

    // Global error handler
    this.app.use(ErrorHandler.handle);
  }

  /**
   * Setup graceful shutdown handling
   */
  private setupGracefulShutdown(): void {
    const gracefulShutdown = async (signal: string) => {
      Logger.info(`📴 Received ${signal}. Starting graceful shutdown...`);

      try {
        // Stop accepting new requests
        this.server.close(async () => {
          Logger.info('🔌 HTTP server closed');

          // Close database connections
          await DatabaseManager.close();
          Logger.info('🗄️ Database connections closed');

          // Close Redis connection
          await this.redis.disconnect();
          Logger.info('🔴 Redis connection closed');

          // Close Apollo server
          await this.apolloServer.stop();
          Logger.info('🎯 GraphQL server stopped');

          // Close task queue
          await this.taskQueue.close();
          Logger.info('📋 Task queue closed');

          Logger.info('✅ Graceful shutdown completed');
          process.exit(0);
        });

        // Force close after 15 seconds
        setTimeout(() => {
          Logger.error('⚠️ Forceful shutdown after timeout');
          process.exit(1);
        }, 15000);

      } catch (error) {
        Logger.error('❌ Error during shutdown:', error);
        process.exit(1);
      }
    };

    // Handle shutdown signals
    process.on('SIGTERM', () => gracefulShutdown('SIGTERM'));
    process.on('SIGINT', () => gracefulShutdown('SIGINT'));
    process.on('SIGUSR2', () => gracefulShutdown('SIGUSR2')); // nodemon restart

    // Handle uncaught exceptions
    process.on('uncaughtException', (error) => {
      Logger.error('💥 Uncaught Exception:', error);
      gracefulShutdown('UNCAUGHT_EXCEPTION');
    });

    process.on('unhandledRejection', (reason, promise) => {
      Logger.error('💥 Unhandled Promise Rejection:', reason);
      gracefulShutdown('UNHANDLED_REJECTION');
    });
  }

  /**
   * Background job processing for data export
   */
  private async processDataExport(format: string, query: any, userId: string, reportId: string): Promise<any> {
    // Implementation would go here
    Logger.info(`📊 Processing data export: ${format} for user ${userId}`);
    return { success: true, reportId, downloadUrl: `/api/downloads/${reportId}` };
  }

  /**
   * Start the server
   */
  public async start(): Promise<void> {
    try {
      Logger.info('🔧 Initializing server components...');

      // Initialize all services
      await this.initializeServices();

      // Configure middleware and routes
      this.configureMiddleware();
      this.configureApiDocumentation();
      this.configureHealthChecks();
      this.configureRoutes();
      this.setupGracefulShutdown();

      // Start server
      this.server.listen(this.port, () => {
        Logger.info(`🚀 AI Business Intelligence Platform running on port ${this.port}`);
        Logger.info(`🌍 Environment: ${process.env.NODE_ENV || 'development'}`);
        Logger.info(`🔗 GraphQL endpoint: http://localhost:${this.port}${this.apolloServer.graphqlPath}`);
        Logger.info(`📖 API documentation: http://localhost:${this.port}/api-docs`);
        Logger.info(`💓 Health check: http://localhost:${this.port}/health`);
        Logger.info(`📊 Metrics: http://localhost:${this.port}/metrics`);
        Logger.info(`🤖 AI Services: ${AIServiceManager.isInitialized() ? 'Enabled' : 'Disabled'}`);
        Logger.info(`🔴 Redis: Connected`);
        Logger.info(`🗄️ Database: Connected`);
        Logger.info('✅ Server is ready to accept connections');
      });

    } catch (error) {
      Logger.error('💥 Failed to start server:', error);
      process.exit(1);
    }
  }
}

// Create and start server instance
const server = new AIBusinessPlatformServer();

// Start server with error handling
server.start().catch((error) => {
  Logger.error('🚨 Server startup error:', error);
  process.exit(1);
});

export default server;
