variable "aws_region" {
  type        = string
  default     = "us-east-1"
  description = "AWS Region to deploy resource into"
}

variable "environment" {
  type        = string
  default     = "production"
  description = "Deployment environment name"
}

variable "app_name" {
  type        = string
  default     = "fraud-detector-api"
  description = "Application name for tagging and resource naming"
}

variable "container_port" {
  type        = number
  default     = 8000
  description = "FastAPI server port"
}

variable "cpu_units" {
  type        = number
  default     = 512
  description = "Fargate CPU units (512 = 0.5 vCPU)"
}

variable "memory_limit" {
  type        = number
  default     = 1024
  description = "Fargate Memory limit in MiB (1024 = 1 GB)"
}

variable "api_keys" {
  type        = string
  sensitive   = true
  description = "Comma-separated list of valid API keys for authentication"
}
