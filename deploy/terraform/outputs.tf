output "alb_dns_name" {
  value       = aws_alb.main.dns_name
  description = "The public DNS URL of the Application Load Balancer to query the API service"
}
